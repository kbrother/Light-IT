import argparse
import torch
from data import irregular_tensor
from collections import Counter
from huffman import huffman_encoding 

def encoding(_tensor):
    _tensor.rows = _tensor.rows.numpy().tolist()
    _tensor.cols = _tensor.cols.numpy().tolist()
    _tensor.heights = _tensor.heights.numpy().tolist()
    _index = _tensor.rows + _tensor.cols + _tensor.heights            
    
    result_dict = huffman_encoding(_index)
    num_bits = 0
    for index in _index:        
        num_bits += len(result_dict[index])
    
    print(len(_tensor.rows))
    num_params = len(_tensor.rows) + num_bits / 64
    print(f'num params: {num_params}')
    

# python coo.py -tp ../data/23-Irregular-Tensor/cms.npy
# python coo.py -tp ../data/23-Irregular-Tensor/mimic3.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', "--tensor_path", type=str)    
    args = parser.parse_args()
    _tensor = irregular_tensor(args.tensor_path, torch.device("cpu"), False)
    encoding(_tensor)
    