import numpy as np
import torch

class irregular_tensor:
    
    def __init__(self, input_path, device):
        # Initialize variable
        self.src_tensor = np.load(input_path, allow_pickle=True)
        self.k = len(self.src_tensor)
        self.j = self.src_tensor[0].shape[1]
        self.i = np.array([self.src_tensor[_i].shape[0] for _i in range(self.k)])
        self.i_max = np.max(self.i)
                        
        # upload data to gpu
        self.row, self.col, self.height, self.vals = [], [], [], []
        self.num_nnz = 0
        for _k in range(self.k):
            _row, _col = self.src_tensor[_k].nonzero()
            _vals = self.src_tensor[_k][_row, _col]
            _vals = np.asarray(_vals).squeeze()             
            self.row = self.row + _row.tolist()
            self.col = self.col + _col.tolist()
            self.vals = self.vals + _vals.tolist()
            self.height = self.height + [_k] * _row.size
            self.num_nnz += self.src_tensor[_k].count_nonzero()
            
        self.rows = torch.tensor(self.row, device=device, dtype=torch.long)
        self.cols = torch.tensor(self.col, device=device, dtype=torch.long)    
        self.vals = torch.tensor(self.vals, device=device)
        self.heights = torch.tensor(self.height, device=device, dtype=torch.long)                
        self.sq_sum = torch.sum(torch.square(self.vals)).item()