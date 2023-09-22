import numpy as np
import torch
import itertools

class irregular_tensor:
    
    def __init__(self, input_path, device, is_dense):
        # Initialize variable
        self.src_tensor = np.load(input_path, allow_pickle=True)
        self.k = len(self.src_tensor)
        self.j = self.src_tensor[0].shape[1]
        self.i = np.array([self.src_tensor[_i].shape[0] for _i in range(self.k)])
        self.i_max = np.max(self.i)
        
        self.sq_sum = 0
        if is_dense:
            print(is_dense)
            for i in range(self.k):
                self.sq_sum += np.sum(np.square(self.src_tensor[i]))

        # upload data to gpu        
        if not is_dense:
            self.rows_list, self.cols_list, self.heights_list, self.vals_list = [], [], [], []
            self.num_nnz = 0
            for _k in range(self.k):
                _rows, _cols = self.src_tensor[_k].nonzero()
                _vals = self.src_tensor[_k][_rows, _cols]
                if _vals.size > 1:
                    _vals = np.asarray(_vals).squeeze()                          
                self.rows_list.append(_rows.tolist())            
                self.cols_list.append(_cols.tolist())
                if _vals.size == 1:                
                    self.vals_list.append([float(_vals)])
                else:
                    self.vals_list.append(_vals.tolist())
                self.heights_list.append([_k] * _rows.size)  
                self.num_nnz += self.src_tensor[_k].count_nonzero()

            self.rows = list(itertools.chain.from_iterable(self.rows_list))
            self.cols = list(itertools.chain.from_iterable(self.cols_list))
            self.heights = list(itertools.chain.from_iterable(self.heights_list))
            self.vals = list(itertools.chain.from_iterable(self.vals_list))

            self.rows = torch.tensor(self.rows, device=device, dtype=torch.long)
            self.cols = torch.tensor(self.cols, device=device, dtype=torch.long)    
            self.vals = torch.tensor(self.vals, device=device)
            self.heights = torch.tensor(self.heights, device=device, dtype=torch.long)                
            self.sq_sum = torch.sum(torch.square(self.vals)).item()