import argparse
from parafac2 import parafac2
from data import irregular_tensor
import torch
from collections import Counter
import itertools

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
    
# python huffman.py -tp ../data/23-Irregular-Tensor/ml-1m.npy -rp results/ml-1m-rank10.pt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', "--tensor_path", type=str)    
    parser.add_argument('-rp', "--result_path", type=str)   
    args = parser.parse_args()    
        
    _tensor = irregular_tensor(args.tensor_path, torch.device('cpu'))
    result_dict = torch.load(args.result_path)
    print("load finish")
    
    cluster_result = result_dict['mapping'].cpu()  # k x i_max
    print(f'num params: {huffman_encoding(_tensor, cluster_result)/64}')