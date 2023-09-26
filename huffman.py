import argparse
from parafac2 import parafac2
from data import irregular_tensor
import torch
from collections import Counter
import itertools
import math

class tree:
    def __init__(self, entry_id, count):
        self.entry_id = entry_id
        self.count = count
        self.childs = []

def dfs(curr_node, curr_bit, result_dict = {}):
    if len(curr_node.childs) == 0:
        assert(curr_node.entry_id > -1)
        result_dict[curr_node.entry_id] = curr_bit
    else:
        dfs(curr_node.childs[0], curr_bit + [0], result_dict)
        dfs(curr_node.childs[1], curr_bit + [1], result_dict)
    
'''
    cluster_result: k x i_max
    data_rows: k
'''
def huffman_encoding(_tensor, cluster_result):
    cluster_result = cluster_result.numpy().tolist()
    total_map = []
    for i in range(_tensor.k):
        curr_map = [cluster_result[i][j] for j in range(_tensor.i[i])]
        total_map.append(curr_map)
    
    total_map = list(itertools.chain.from_iterable(total_map))
    count_result = Counter(total_map)
    count_result = [tree(k, v) for k, v in sorted(count_result.items(), key=lambda item: item[1])]
    
    # Build huffman trees
    while len(count_result) > 1:
        left_tree = count_result.pop(0)
        right_tree = count_result.pop(0)
        new_tree = tree(-1, left_tree.count + right_tree.count)
        new_tree.childs = [left_tree, right_tree]
        
        idx = 0
        while idx < len(count_result) and new_tree.count >= count_result[idx].count:
            idx += 1
        
        if len(count_result) == idx:
            count_result.append(new_tree)
        else:
            count_result.insert(idx, new_tree)
        
    # DFS to get the bits of each integer
    result_dict = {}
    dfs(count_result[0], [], result_dict)
    
    num_bits = 0
    for i in range(_tensor.k):
        for j in range(_tensor.i[i]):
            num_bits += len(result_dict[cluster_result[i][j]])
                                    
    return num_bits
    
# python huffman.py -tp ../data/23-Irregular-Tensor/cms.npy -rp results/cms-lr0.01-rank5.pt -r 5 -de 0 -d False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', "--tensor_path", type=str)    
    parser.add_argument('-rp', "--result_path", type=str)       
    parser.add_argument('-r', "--rank", type=int)  
    parser.add_argument('-fp', "--factor_path", type=str)            
    parser.add_argument('-d', "--is_dense", type=str, default="error")
    parser.add_argument(
        "-de", "--device",
        action="store", type=int
    )    
    parser.add_argument(
        "-bif", "--batch_init_factor",
        action="store", default=2**6, type=int
    )
    
    parser.add_argument(
        "-cb", "--cluster_batch",
        action="store", default=64, type=int
    )
        
    parser.add_argument(
        "-b", "--batch_size",
        action="store", default=2**7, type=int
    )
    
    parser.add_argument(
        "-tbz", "--tucker_batch_lossz",
        action="store", default=2**6, type=int
    )
    
    parser.add_argument(
        "-tbn", "--tucker_batch_lossnz",
        action="store", default=2**5, type=int
    )
    
    args = parser.parse_args()    
    if args.is_dense == "True":
        args.is_dense = True
    elif args.is_dense == "False":
        args.is_dense = False
    else:
        assert("wrong input")
    
    if args.device == None:
        device = torch.device("cpu")
    else:
        device = torch.device(f'cuda:{args.device}')
    _tensor = irregular_tensor(args.tensor_path, torch.device(device), args.is_dense)
    result_dict = torch.load(args.result_path)
    print("load finish")
    
    _parafac2 = parafac2(_tensor, device, args)        
    _parafac2.centroids.data.copy_(result_dict['centroids'].to(device))
    _parafac2.U.data.copy_(result_dict['U'].to(device))
    _parafac2.V.data.copy_(result_dict['V'].to(device))
    _parafac2.S.data.copy_(result_dict['S'].to(device))
        
    if 'mapping' in result_dict:
        _parafac2.mapping = result_dict['mapping'].to(device) # k x i_max
        _parafac2.mapping_mask = torch.zeros(_tensor.k, _tensor.i_max, dtype=torch.bool, device=device)   # k x i_max
        for _k in range(_tensor.k):        
            _parafac2.mapping_mask[_k, :_tensor.i[_k]] = True
        _parafac2.G = result_dict['G'].to(device)    
        with torch.no_grad():
            if args.is_dense:
                sq_loss = _parafac2.L2_loss_tucker_dense(args.tucker_batch_loss_nz)
            else:
                sq_loss = _parafac2.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)                
            print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(_tensor.sq_sum)}')
        cluster_result = result_dict['mapping'].cpu()  # k x i_max
    else:
        with torch.no_grad():
            if args.is_dense:
                sq_loss = _parafac2.L2_loss_dense(False, args.batch_size, _parafac2.U * _parafac2.U_mask)
            else:
                sq_loss = _parafac2.L2_loss(False, args.batch_size, _parafac2.U * _parafac2.U_mask)
            print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(_tensor.sq_sum)}')
        cluster_result = _parafac2.clustering(args).cpu()  # k x i_max
        
    print(f'num params: {huffman_encoding(_tensor, cluster_result)/64}')