from scipy.io import loadmat
from data import irregular_tensor
import math
import torch
from tqdm import tqdm
import numpy as np
import gc
import itertools

def clear_memory():
    gc.collect()  
    torch.cuda.empty_cache()
    
# input1: k x row1 x col1
# input2: k x row2 x col2
def batch_kron(input1, input2):
    num_row1, num_col1 = input1.shape[1], input1.shape[2]
    num_row2, num_col2 = input2.shape[1], input2.shape[2]    
    
    input1 = torch.repeat_interleave(input1, num_row2, dim=1)
    input1 = torch.repeat_interleave(input1, num_col2, dim=2)  # k x row1*row2 x col1*col2
    input2 = input2.repeat(1, num_row1, 1)    
    input2 = input2.repeat(1, 1, num_col1)    
    return input1 * input2


# input1: row1 x col
# input2: row2 x col
def khatri_rao(input1, input2):
    num_row1, num_row2 = input1.shape[0], input2.shape[0]    
    input1 = torch.repeat_interleave(input1, num_row2, dim=0)   # row1*row2 x col
    input2 = input2.repeat(num_row1, 1)   # row1*row2 x col
    return input1 * input2
    
    
# input1: row x col1
# input2: row x col2
def face_split(input1, input2):
    num_col1, num_col2 = input1.shape[1], input2.shape[1]   
    input1 = torch.repeat_interleave(input1, num_col2, dim=1)    # row x col1*col2    
    input2 = input2.repeat(1, num_col1)
    return input1 * input2

    
class parafac2:                    
    def __init__(self, _tensor, device, args):      
        # Intialization
        self.device = device
        self.tensor = _tensor
        if args.is_dense:
            scale_factor = 0.1
        else:
            scale_factor = 0.01
        
        _sum = 0
        self.U_sidx = [0]  # num_tensor + 1, idx of tensor slice -> row idx of U
        self.U_mapping = []  # i_sum, row idx of U -> idx of tensor slice        
        self.mapping = []
        for i in range(_tensor.num_tensor):
            _sum += _tensor.first_dim[i]
            self.U_sidx.append(_sum)
            for j in range(_tensor.first_dim[i]):
                self.U_mapping.append(i)

            self.mapping.append(list(range(_tensor.first_dim[i])))

        self.mapping = list(itertools.chain.from_iterable(self.mapping))
        self.mapping = torch.tensor(self.mapping, device=self.device, dtype=torch.long)
        self.U_sidx = torch.tensor(self.U_sidx, device=self.device, dtype=torch.long)
        self.U_mapping = torch.tensor(self.U_mapping, device=self.device, dtype=torch.long)
        self.num_first_dim = self.U_mapping.shape[0]
        self.rank = args.rank     
        self.V = []
        for m in range(self.tensor.order-2):
            curr_dim = _tensor.middle_dim[m]
            self.V.append(scale_factor * torch.rand((curr_dim, self.rank), device=device, dtype=torch.double))  # j x rank
        self.S = scale_factor * torch.rand((_tensor.num_tensor, self.rank), device=device, dtype=torch.double)      # k x rank             

        # Upload to gpu        
        self.centroids = scale_factor * torch.rand((_tensor.max_first, self.rank), device=device, dtype=torch.double)    # cluster centers,  i_max x rank
        self.centroids = torch.nn.Parameter(self.centroids)                      
        self.S = torch.nn.Parameter(self.S)
        for m in range(_tensor.order-2):
            self.V[m] = torch.nn.Parameter(self.V[m])
        
        with torch.no_grad():
            if args.is_dense:
                sq_loss = self.L2_loss_dense(args, False)
            else:
                sq_loss = self.L2_loss(args, False)
            print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)}') 
            print(f'square loss: {sq_loss}')       
    
    '''
        Return a tensor of size 'batch size' x j_1*j_2*...*j_(d-2)        
    '''
    def set_curr_tensor(self, batch_size, start_k):
        start_idx = self.U_sidx[start_k].item()
        end_idx = self.U_sidx[start_k + batch_size].item()
        curr_tensor = self.tensor.src_tensor_torch[start_idx:end_idx].to(self.device)  # bs' x i_2 x ... x i_(m-1)  
        curr_tensor = torch.reshape(curr_tensor, (end_idx - start_idx, -1))   # bs' x i_2 * ... * i_(m-1) 
        return curr_tensor
        
        
    '''
        Return a tensor of size 'batch size' x j_1*j_2*...*j_(d-2)        
    '''
    def set_curr_tensor_new(self, batch_size, start_i):        
        curr_tensor = self.tensor.src_tensor_torch[start_i:start_i + batch_size].to(self.device)  # bs' x i_2 x ... x i_(m-1)  
        curr_tensor = torch.reshape(curr_tensor, (batch_size, -1))   # bs' x i_2 * ... * i_(m-1) 
        return curr_tensor
    
    
    '''
        input_U: num_tensor x i_max x rank
    '''
    def L2_loss_dense(self, args, is_train):
        _loss = 0            
        for i in tqdm(range(0, self.num_first_dim, args.batch_lossnz)):                        
            Vprod = self.V[0]
            for j in range(1, self.tensor.order-2):
                Vprod = khatri_rao(Vprod, self.V[j])   # i_2 * ... * i_(m-1) x rank        
       
            curr_batch_size = min(args.batch_lossnz, self.num_first_dim - i) 
            assert(curr_batch_size > 1)
            s_idx = self.U_mapping[i:i+curr_batch_size]
            curr_S = self.S[s_idx, :].unsqueeze(1)  # bs' x 1 x rank
            VS = Vprod.unsqueeze(0) * curr_S   # bs' x i_2 * ... * i_(m-1) x rank        
            VS = torch.transpose(VS, 1, 2)   # bs' x rank x i_2 * ... * i_(m-1)        
            
            curr_mapping = self.mapping[i:i+curr_batch_size]   # bs'
            curr_U = self.centroids[curr_mapping]  # bs' x rank
                
             # curr_U: batch size x i_max x rank
            approx = torch.bmm(curr_U.unsqueeze(1), VS).squeeze()   # bs' x i_2 * ... * i_(m-1)                    
            curr_tensor = self.set_curr_tensor_new(curr_batch_size, i)  # bs' x i_2 * ... * i_(m-1)                                
            #curr_loss = torch.sum(torch.square(approx))
            curr_loss = torch.sum(torch.square(approx - curr_tensor))
            if is_train:
                curr_loss.backward()
            _loss += curr_loss.item()
        
        return _loss

    
    '''
        mode: train or test or parafac2
    '''
    def L2_loss(self, args, is_train):                    
        _loss = 0
        for i in tqdm(range(0, self.tensor.num_tensor, args.batch_lossz)):
            # zero terms             
            VtV = torch.ones((self.rank, self.rank), device=self.device, dtype=torch.double)  # r x r
            for j in range(self.tensor.order-2):
                VtV = VtV * (self.V[j].t() @ self.V[j])  # r x r
            
            curr_batch_size = min(args.batch_lossz, self.tensor.num_tensor - i)
            assert(curr_batch_size > 1)                                
            curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]  # bs'
            curr_U = self.centroids[curr_mapping]  # bs' x rank
            
            UtU_input = torch.bmm(curr_U.unsqueeze(-1), curr_U.unsqueeze(1))   # bs' x rank x rank                
            UtU = torch.zeros((curr_batch_size, self.rank, self.rank), device=self.device, dtype=torch.double)
            temp_idx = self.U_mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]] - i
            #print(temp_idx.shape)
            #print(UtU_input.shape)
            UtU = UtU.index_add_(0, temp_idx, UtU_input)   # k x rank x rank                
            
            curr_S = self.S[i:i+curr_batch_size, :]
            StS = torch.bmm(curr_S.unsqueeze(2), curr_S.unsqueeze(1)) # k x rank x rank
            first_mat = torch.sum(UtU * StS, dim=0)  # rank x rank
            sq_sum = torch.sum(first_mat * VtV)
            if is_train: 
                sq_sum.backward()
            _loss += sq_sum.item()  
    
        # Correct non-zero terms                
        for i in tqdm(range(0, self.tensor.num_nnz, args.batch_lossnz)):
            curr_batch_size = min(args.batch_lossnz, self.tensor.num_nnz - i)
            assert(curr_batch_size > 1)
            first_idx = torch.tensor(self.tensor.indices[0][i: i+curr_batch_size], device=self.device, dtype=torch.long) # bs
            final_idx = torch.tensor(self.tensor.indices[-1][i: i+curr_batch_size], device=self.device, dtype=torch.long)  # bs
            
            first_idx = first_idx + self.U_sidx[final_idx]   # batch size            
            curr_mapping = self.mapping[first_idx]                
            curr_U = self.centroids[curr_mapping]

            approx = curr_U * self.S[final_idx, :]  # bs x rank
            for m in range(1, self.tensor.order-1):
                curr_idx = torch.tensor(self.tensor.indices[m][i: i+curr_batch_size], device=self.device, dtype=torch.long)
                approx = approx * self.V[m-1][curr_idx, :]   # bs x rank
            
            curr_value = torch.tensor(self.tensor.values[i: i+curr_batch_size], dtype=torch.double, device=self.device)
            approx = torch.sum(approx, dim=1)
            #sq_err = -torch.sum(torch.square(approx))            
            sq_err = torch.sum(torch.square(curr_value - approx) - torch.square(approx))            
            
            if is_train: sq_err.backward()
            _loss += sq_err.item()
            
        return _loss

        
    def quantization(self, args):
        optimizer = torch.optim.Adam([self.S, self.centroids] + self.V, lr=args.lr)
        max_fitness = -100
        for _epoch in range(args.epoch):
            optimizer.zero_grad()
            if args.is_dense:
                self.L2_loss_dense(args, True) 
            else:
                self.L2_loss(args, True)
                           
            #del curr_mapping, curr_U, curr_U_cluster
            #clear_memory()            
            optimizer.step()            
            if (_epoch + 1) % 10 == 0:
                with torch.no_grad():
                    if args.is_dense:
                        _loss = self.L2_loss_dense(args, False) 
                    else:
                        _loss = self.L2_loss(args, False)              
                    _fitness = 1 - math.sqrt(_loss)/math.sqrt(self.tensor.sq_sum)
                    print(f'epoch: {_epoch}, l2 loss: {_loss}, fitness: {_fitness}')
                    with open(args.output_path + ".txt", 'a') as f:
                        f.write(f'epoch: {_epoch}, l2 loss: {_loss}, fitness: {_fitness}\n')
                       
                    if _fitness > max_fitness:
                        max_fitness = _fitness
                        final_cents = self.centroids.data.clone().detach().cpu()
                        final_V = [_v.data.clone().detach().cpu() for _v in self.V]                        
                        final_S = self.S.data.clone().detach().cpu()                        
                        
        self.centroids.data.copy_(final_cents.to(self.device))        
        
        for m in range(self.tensor.order-2):
            self.V[m].data.copy_(final_V[m].to(self.device))
        self.S.data.copy_(final_S.to(self.device)) 
        
        torch.save({
            'fitness': max_fitness, 'centroids': self.centroids.data,
            'S': self.S.data, 'V': final_V,
        }, args.output_path + "_cp.pt")                