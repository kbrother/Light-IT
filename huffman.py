import argparse
from parafac2 import parafac2
from data import irregular_tensor
import torch
from collections import Counter
import itertools
import math
import heapq

class tree:
    def __init__(self, entry_id, count):
        self.entry_id = entry_id
        self.count = count
        self.childs = []
    
    # override the comparison operator
    def __lt__(self, nxt):
        return self.count < nxt.count


def dfs(curr_node, curr_bit, result_dict = {}):
    if len(curr_node.childs) == 0:
        assert(curr_node.entry_id > -1)
        result_dict[curr_node.entry_id] = curr_bit
    else:
        dfs(curr_node.childs[0], curr_bit + [0], result_dict)
        dfs(curr_node.childs[1], curr_bit + [1], result_dict)
    
    
def huffman_encoding(indices):
    count_result = Counter(indices)
    count_result = [tree(k, v) for k, v in sorted(count_result.items(), key=lambda item: item[1])]
    heapq.heapify(count_result)
    
    # Build huffman trees
    while len(count_result) > 1:
        left_tree = heapq.heappop(count_result)
        right_tree = heapq.heappop(count_result)
        new_tree = tree(-1, left_tree.count + right_tree.count)
        new_tree.childs = [left_tree, right_tree]        
        heapq.heappush(count_result, new_tree)
        
    # DFS to get the bits of each integer
    result_dict = {}
    dfs(count_result[0], [], result_dict)    
    return result_dict
    
    
'''
    cluster_result: k x i_max
    data_rows: k
'''
def encoding(_tensor, cluster_result):
    cluster_result = cluster_result.numpy().tolist()
    result_dict = huffman_encoding(cluster_result)
    
    num_bits = 0
    for i in range(len(cluster_result)):        
        num_bits += len(result_dict[cluster_result[i]])
                                    
    return num_bits
    
    
# python huffman.py -tp ../data/23-Irregular-Tensor/cms.npy -rp results/cms-lr0.01-rank5.pt -r 5 -de 0 -d False
# python huffman.py -tp ../data/23-Irregular-Tensor/mimic3.npy -rp results/mimic3-lr0.01-rank5.pt -r 5 -de 4 -d False
# python huffman.py -tp ../input/23-Irregular-Tensor/delicious.pickle -rp results/delicious_r5_lr0.01.pt -r 5 -de 0 -d False -cp True
# python huffman.py -tp ../data/23-Irregular-Tensor/enron.pickle -rp results/enron_r5_lr0.01_cp.pt -r 5 -de 6 -d False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', "--tensor_path", type=str)    
    parser.add_argument('-rp', "--result_path", type=str)       
    parser.add_argument('-r', "--rank", type=int)       
    parser.add_argument('-cp', "--is_cp", type=str)
    parser.add_argument('-d', "--is_dense", type=str, default="error")
    parser.add_argument(
        "-de", "--device",
        action="store", type=int
    )    

    parser.add_argument(
        "-bz", "--batch_lossz",
        action="store", default=2**10, type=int
    )
    
    parser.add_argument(
        "-bnz", "--batch_lossnz",
        action="store", default=2**22, type=int
    )
    
    parser.add_argument(
        "-cb", "--cluster_batch",
        action="store", default=128, type=int
    )
    
    parser.add_argument(
        "-tbz", "--tucker_batch_lossz",
        action="store", default=2**10, type=int
    )
    
    parser.add_argument(
        "-tbnz", "--tucker_batch_lossnz",
        action="store", default=2**10, type=int
    )
    
    parser.add_argument(
        "-tbnx", "--tucker_batch_alsnx",
        action="store", default=2**10, type=int
    )
    
    args = parser.parse_args()    
    if args.is_dense == "True":
        args.is_dense = True
    elif args.is_dense == "False":
        args.is_dense = False
    else:
        assert("wrong input")
        
    if args.is_cp == "True":
        args.is_cp= True
    else:
        args.is_cp =False
    
    if args.device == None:
        device = torch.device("cpu")
    else:
        device = torch.device(f'cuda:{args.device}')
    _tensor = irregular_tensor(args.tensor_path, args.is_dense)
    result_dict = torch.load(args.result_path)
    print("load finish")
    
    _parafac2 = parafac2(_tensor, device, False, args)        
    _parafac2.centroids.data.copy_(result_dict['centroids'].to(device))
    for m in range(_tensor.order-2):
        _parafac2.V[m].data.copy_(result_dict['V'][m].to(device))
    _parafac2.S.data.copy_(result_dict['S'].to(device))
        
    _parafac2.mapping = result_dict['mapping'].to(device) # k x i_max
    if not args.is_cp:
        _parafac2.G = result_dict['G'].to(device)    
    with torch.no_grad():
        if args.is_dense:
            if args.is_cp:
                with toch.no_grad():
                    sq_loss = _parafac2.L2_loss_dense(args, False, "test")
            else:
                sq_loss = _parafac2.L2_loss_tucker_dense(args.tucker_batch_lossnz)
        else:
            if args.is_cp:
                with torch.no_grad():
                    sq_loss = _parafac2.L2_loss(args, False, "test")
            else:
                sq_loss = _parafac2.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)                
        print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(_tensor.sq_sum)}')
    cluster_result = result_dict['mapping'].cpu()  # k x i_max
        
    num_bytes = torch.numel(_parafac2.centroids)
    for m in range(_tensor.order-2):
        num_bytes += torch.numel(_parafac2.V[m])
    num_bytes += torch.numel(_parafac2.S)  
    num_bytes *= 8
    num_bytes += encoding(_tensor, cluster_result)/8 
    print(f'num bytes: {num_bytes}')