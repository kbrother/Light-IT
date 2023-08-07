import numpy as np
import torch
import itertools

class irregular_tensor:
    
    def __init__(self, input_path, device):
        # Initialize variable
        self.src_tensor = np.load(input_path, allow_pickle=True)
        self.k = len(self.src_tensor)
        self.j = self.src_tensor[0].shape[1]
        self.i = np.array([self.src_tensor[_i].shape[0] for _i in range(self.k)])
        self.i_max = np.max(self.i)
                        
        # upload data to gpu
        self.rows, self.cols, self.heights, self.vals = [], [], [], []
        self.num_nnz = 0
        for _k in range(self.k):
            _rows, _cols = self.src_tensor[_k].nonzero()
            _vals = self.src_tensor[_k][_rows, _cols]
            if _vals.size > 1:
                _vals = np.asarray(_vals).squeeze()                          
            self.rows.append(_rows.tolist())            
            self.cols.append(_cols.tolist())
            if _vals.size == 1:                
                self.vals.append([_vals[0]])
            else:
                self.vals.append(_vals.tolist())
            self.heights.append([_k] * _rows.size)  
            self.num_nnz += self.src_tensor[_k].count_nonzero()
            
        self.rows = list(itertools.chain.from_iterable(self.rows))
        self.cols = list(itertools.chain.from_iterable(self.cols))
        self.heights = list(itertools.chain.from_iterable(self.heights))
        self.vals = list(itertools.chain.from_iterable(self.vals))
        
        self.rows = torch.tensor(self.rows, device=device, dtype=torch.long)
        self.cols = torch.tensor(self.cols, device=device, dtype=torch.long)    
        self.vals = torch.tensor(self.vals, device=device)
        self.heights = torch.tensor(self.heights, device=device, dtype=torch.long)                
        self.sq_sum = torch.sum(torch.square(self.vals)).item()