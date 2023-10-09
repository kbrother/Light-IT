import numpy as np
import torch
import itertools
import pickle

class irregular_tensor:
    
    def __init__(self, input_path, is_dense):        
        self.sq_sum = 0
        if is_dense:
            # Initialize variable
            self.src_tensor = np.load(input_path, allow_pickle=True)
            self.num_tensor = len(self.src_tensor)
            self.mode = len(self.src_tensor[0].shape) + 1
            self.middle_dim = self.src_tensor[0].shape[1:] 
            self.first_dim = np.array([self.src_tensor[_i].shape[0] for _i in range(self.num_tensor)])
            self.max_first = np.max(self.first_dim)

            for i in range(self.num_tensor):
                self.sq_sum += np.sum(np.square(self.src_tensor[i]))

            temp_dims = [self.num_tensor, self.max_first] + list(self.middle_dim)
            self.src_tensor_torch = torch.zeros(temp_dims, dtype=torch.double)
            for i in range(self.num_tensor):
                self.src_tensor_torch[i,:self.first_dim[i],:] = torch.tensor(self.src_tensor[i], dtype=torch.double)
                
        # upload data to gpu        
        if not is_dense:
            with open(input_path, 'rb') as f:
                raw_dict = pickle.load(f)            
            self.indices = raw_dict['idx']
            self.num_nnz = len(self.indices[0])
            self.values = np.array(raw_dict['val'])
            self.mode = len(self.indices)
            
            # sort indices            
            for m in range(self.mode):
                self.indices[m] = np.array(self.indices[m])
            
            idx2newidx = np.argsort(self.indices[-1])
            for m in range(self.mode):
                self.indices[m] = self.indices[m][idx2newidx]
            self.values = self.values[idx2newidx]            
            
            # save tensor stat        
            self.max_first = max(self.indices[0]) + 1
            self.num_tensor = max(self.indices[-1]) + 1
            self.middle_dim = []
            for m in range(1, self.mode-1):
                self.middle_dim.append(max(self.indices[m]) + 1)                                                                                                      
            self.sq_sum = np.sum(self.values**2)            
            self.tidx2start = [0]
            for i in range(self.num_nnz):
                if self.indices[-1][self.tidx2start[-1]] != self.indices[-1][i]:
                    self.tidx2start.append(i)
            self.tidx2start.append(self.num_nnz)
            
            # Set first dim
            self.first_dim = []
            for i in range(self.num_tensor):
                self.first_dim.append(max(self.indices[0][self.tidx2start[i]:self.tidx2start[i+1]]) + 1)
            self.first_dim = np.array(self.first_dim)