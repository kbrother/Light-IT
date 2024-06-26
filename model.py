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

    
class LightIT:        
    def init_factor(self, is_dense):
        with torch.no_grad():
            _H = torch.rand((self.rank, self.rank), device=self.device, dtype=torch.double)        
            if is_dense:
                Vprod = self.V[0]
                for m in range(1, self.tensor.order-2):
                    Vprod = khatri_rao(Vprod, self.V[m])   # row2 * rows3 *.... x rank
                
            for i in tqdm(range(self.tensor.num_tensor)):                                
                # build tensor
                if is_dense:                      
                    curr_dims = [self.tensor.first_dim[i]] + list(self.tensor.middle_dim)
                    curr_tensor = torch.from_numpy(self.tensor.src_tensor[i]).to(self.device)
                    curr_tensor = torch.reshape(curr_tensor, (self.tensor.first_dim[i], -1))  # i_max x ...       
                
                    VS = Vprod * self.S[i].unsqueeze(0) # row2 * rows3 *.... x rank
                    XVS = curr_tensor @ VS  # i_max x R                     
                else:   
                    curr_tidx = self.tensor.tidx2start[i]
                    next_tidx = self.tensor.tidx2start[i + 1]
                    _V = torch.ones((next_tidx - curr_tidx, self.rank), device=self.device, dtype=torch.double)   # curr nnz x rank
                    for m in range(self.tensor.order - 2):
                        curr_idx = torch.tensor(self.tensor.indices[m + 1][curr_tidx:next_tidx], dtype=torch.long, device=self.device)
                        _V = _V * self.V[m][curr_idx, :]
                    
                    curr_idx = torch.tensor(self.tensor.indices[self.tensor.order - 1][curr_tidx:next_tidx], dtype=torch.long, device=self.device)
                    VS = _V * self.S[curr_idx, :]   # curr nnz x rank
                    curr_X = torch.tensor(self.tensor.values[curr_tidx:next_tidx], device=self.device, dtype=torch.double)
                    XVS_raw = curr_X.unsqueeze(1) * VS  # curr nnz x rank                
                    XVS = torch.zeros((self.tensor.first_dim[i], self.rank), device=self.device, dtype=torch.double)  # i_curr x rank
                    
                    curr_idx = torch.tensor(self.tensor.indices[0][curr_tidx:next_tidx], dtype=torch.long, device=self.device)    # nnz
                    XVS = XVS.index_add_(0, curr_idx, XVS_raw)  # i_curr x rank                    
                
                # compute SVD                               
                XVSH = XVS @ _H.t()   # i_curr x R                                 
                Z, Sigma, Ph = torch.linalg.svd(XVSH, full_matrices=False)  # Z: i_max x R, Ph:  R x R
                self.U[self.U_sidx[i].item():self.U_sidx[i+1].item(),:] = torch.mm(Z, Ph) # i_max x rank

            # Normalize entry 
            _lambda = torch.sqrt(torch.sum(torch.square(_H), dim=0))  # R
            _H = _H / _lambda.unsqueeze(0)
            
            for m in range(self.tensor.order-2):
                curr_lambda = torch.sqrt(torch.sum(torch.square(self.V[m]), dim=0))
                _lambda = _lambda * curr_lambda
                self.V[m] = self.V[m] / curr_lambda.unsqueeze(0)
                
            curr_lambda = torch.sqrt(torch.sum(torch.square(self.S), dim=0))
            _lambda = _lambda * curr_lambda
            self.S = self.S / curr_lambda.unsqueeze(0)
            
            _lambda = torch.pow(_lambda, 1/self.tensor.order)
            _H = _H * _lambda.unsqueeze(0)
            for m in range(self.tensor.order-2):
                self.V[m] = self.V[m] * _lambda.unsqueeze(0)
            self.S = self.S * _lambda.unsqueeze(0)
            
            # Normalize U
            self.U = self.U @ _H   # i_sum x rank
        '''    
        UtU_input = torch.bmm(self.U.unsqueeze(-1), self.U.unsqueeze(1))   # i_sum x rank x rank                
        UtU = torch.zeros((self.tensor.num_tensor, self.rank, self.rank), device=self.device, dtype=torch.double)
        UtU = UtU.index_add_(0, self.U_mapping, UtU_input)   # k x rank x rank
        
        HtH = _H.t() @ _H
        #iden = torch.diag(torch.ones(self.rank)).to(self.device)
        print(torch.mean(torch.abs(UtU - HtH.unsqueeze(0))))        
        '''   

    def __init__(self, _tensor, device, require_init, args):      
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
        for i in range(_tensor.num_tensor):
            _sum += _tensor.first_dim[i]
            self.U_sidx.append(_sum)
            for j in range(_tensor.first_dim[i]):
                self.U_mapping.append(i)
        
        self.U_sidx = torch.tensor(self.U_sidx, device=self.device, dtype=torch.long)
        self.U_mapping = torch.tensor(self.U_mapping, device=self.device, dtype=torch.long)
        self.num_first_dim = self.U_mapping.shape[0]
        self.rank = args.rank
        self.U = scale_factor * torch.rand((self.num_first_dim, self.rank), device=device, dtype=torch.double)  # i_sum x rank              
        self.V = []
        for m in range(self.tensor.order-2):
            curr_dim = _tensor.middle_dim[m]
            self.V.append(scale_factor * torch.rand((curr_dim, self.rank), device=device, dtype=torch.double))  # j x rank
        self.S = scale_factor * torch.rand((_tensor.num_tensor, self.rank), device=device, dtype=torch.double)      # k x rank        
        if require_init:
            self.init_factor(args.is_dense)        

        # Upload to gpu        
        self.centroids = scale_factor * torch.rand((_tensor.max_first, self.rank), device=device, dtype=torch.double)    # cluster centers,  i_max x rank
        self.centroids = torch.nn.Parameter(self.centroids)                      
        self.U = torch.nn.Parameter(self.U)
        self.S = torch.nn.Parameter(self.S)
        for m in range(_tensor.order-2):
            self.V[m] = torch.nn.Parameter(self.V[m])
        
        if args.is_dense:
            self.random_idx = np.random.permutation(self.num_first_dim)
            self.shuffled_U_mapping = self.U_mapping[self.random_idx]   
            self.shuffled_tensor = self.tensor.src_tensor_torch[self.random_idx]
            
        else:
            random_idx = np.random.permutation(self.tensor.num_nnz)
            self.shuffled_indices = []
            for m in range(self.tensor.order):
                self.shuffled_indices.append(self.tensor.indices[m][random_idx])
            self.shuffled_vals = self.tensor.values[random_idx]
        
        
        with torch.no_grad():
            if args.is_dense:
                sq_loss = self.L2_loss_dense(args, False, "parafac2")
            else:
                sq_loss = self.L2_loss(args, False, "parafac2")
            print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)}') 
            print(f'square loss: {sq_loss}')       
        
    '''
        Return a tensor of size 'batch size' x j_1*j_2*...*j_(d-2)        
    '''
    def set_curr_tensor_new(self, batch_size, start_i, _order):        
        curr_tensor = self.tensor.src_tensor_torch[start_i:start_i + batch_size].to(self.device)  # bs' x i_2 x ... x i_(m-1)  
        num_rows = 1
        for i in range(_order + 1):
            num_rows *= curr_tensor.shape[i]
        
        curr_tensor = torch.reshape(curr_tensor, (num_rows, -1))   # bs' x i_2 * ... * i_(m-1) 
        return curr_tensor
    
    
    '''
        Return a tensor of size 'batch size' x j_1*j_2*...*j_(d-2)        
    '''
    def set_curr_tensor_shuffled(self, batch_size, start_i, _order):        
        curr_tensor = self.shuffled_tensor[start_i:start_i + batch_size].to(self.device)  # bs' x i_2 x ... x i_(m-1)  
        num_rows = 1
        for i in range(_order + 1):
            num_rows *= curr_tensor.shape[i]
        
        curr_tensor = torch.reshape(curr_tensor, (num_rows, -1))   # bs' x i_2 * ... * i_(m-1) 
        return curr_tensor    
    
    '''
        Return a tensor of size 'batch size' x j_1*j_2*...*j_(d-2)        
    '''
    def set_curr_tensor_custom(self, input_idx, _order):        
        curr_tensor = self.tensor.src_tensor_torch[input_idx].to(self.device)  # bs' x i_2 x ... x i_(m-1)  
        num_rows = 1
        for i in range(_order + 1):
            num_rows *= curr_tensor.shape[i]
        
        curr_tensor = torch.reshape(curr_tensor, (num_rows, -1))   # bs' x i_2 * ... * i_(m-1) 
        return curr_tensor    
    
    
    '''
        input_U: num_tensor x i_max x rank
    '''
    def L2_loss_dense(self, args, is_train, mode):
        _loss = 0            
        for i in tqdm(range(0, self.num_first_dim, args.batch_nz)):                        
            Vprod = self.V[0]
            for j in range(1, self.tensor.order-2):
                Vprod = khatri_rao(Vprod, self.V[j])   # i_2 * ... * i_(m-1) x rank        
       
            curr_batch_size = min(args.batch_nz, self.num_first_dim - i) 
            assert(curr_batch_size > 1)
            #s_idx = self.U_mapping[input_idx]
            s_idx = self.shuffled_U_mapping[i:i+curr_batch_size]
            curr_S = self.S[s_idx, :].unsqueeze(1)  # bs' x 1 x rank
            VS = Vprod.unsqueeze(0) * curr_S   # bs' x i_2 * ... * i_(m-1) x rank        
            VS = torch.transpose(VS, 1, 2)   # bs' x rank x i_2 * ... * i_(m-1)        
            
            temp_random_idx = torch.tensor(self.random_idx[i:i+curr_batch_size], dtype=torch.long, device=self.device)
            curr_U = self.U[temp_random_idx]    # bs' x rank            
            #curr_U = self.U[input_idx]    # bs' x rank
            if mode != "parafac2":
                #curr_mapping = self.mapping[input_idx]   # bs'
                curr_mapping = self.shuffled_mapping[i:i+curr_batch_size]
                curr_U_cluster = self.centroids[curr_mapping]  # bs' x rank
                if mode=="train":                
                    sg_part = (curr_U - curr_U_cluster).detach()
                    curr_U = curr_U - sg_part
                else:
                    curr_U = curr_U_cluster
             # curr_U: batch size x i_max x rank
            approx = torch.bmm(curr_U.unsqueeze(1), VS).squeeze()   # bs' x i_2 * ... * i_(m-1)                    
            curr_tensor = self.set_curr_tensor_shuffled(curr_batch_size, i, 0)  # bs' x i_2 * ... * i_(m-1)                                
            #curr_loss = torch.sum(torch.square(approx))
            curr_loss = torch.sum(torch.square(approx - curr_tensor))
            if is_train:
                curr_loss.backward()
            _loss += curr_loss.item()
        
        return _loss

    
    '''
        mode: train or test or parafac2
    '''
    def L2_loss(self, args, is_train, mode):                    
        _loss = 0
        for i in tqdm(range(0, self.tensor.num_tensor, args.batch)):
            # zero terms             
            VtV = torch.ones((self.rank, self.rank), device=self.device, dtype=torch.double)  # r x r
            for j in range(self.tensor.order-2):
                VtV = VtV * (self.V[j].t() @ self.V[j])  # r x r
            
            curr_batch_size = min(args.batch, self.tensor.num_tensor - i)
            assert(curr_batch_size > 1)            
                        
            curr_U = self.U[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]   # bs' x R
            if mode != "parafac2":
                curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]  # bs'
                curr_U_cluster = self.centroids[curr_mapping]  # bs' x rank
                if mode=="train":                
                    sg_part = (curr_U - curr_U_cluster).detach()
                    curr_U = curr_U - sg_part
                else:
                    curr_U = curr_U_cluster
            
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
        for i in tqdm(range(0, self.tensor.num_nnz, args.batch_nz)):
            curr_batch_size = min(args.batch_nz, self.tensor.num_nnz - i)
            assert(curr_batch_size > 1)
            first_idx = torch.tensor(self.shuffled_indices[0][i:i+curr_batch_size], device=self.device, dtype=torch.long) # bs
            final_idx = torch.tensor(self.shuffled_indices[-1][i:i+curr_batch_size], device=self.device, dtype=torch.long)  # bs
            
            first_idx = first_idx + self.U_sidx[final_idx]   # batch size
            curr_U = self.U[first_idx, :]   # bs' x rank
            if mode != "parafac2":
                curr_mapping = self.mapping[first_idx]
                if mode == "train":                
                    sg_part = (curr_U - self.centroids[curr_mapping]).detach()
                    curr_U = curr_U - sg_part
                else:
                    curr_U = self.centroids[curr_mapping]

            approx = curr_U * self.S[final_idx, :]  # bs x rank
            for m in range(1, self.tensor.order-1):
                curr_idx = torch.tensor(self.shuffled_indices[m][i:i+curr_batch_size], device=self.device, dtype=torch.long)
                approx = approx * self.V[m-1][curr_idx, :]   # bs x rank
            
            curr_value = torch.tensor(self.shuffled_vals[i:i+curr_batch_size], dtype=torch.double, device=self.device)
            approx = torch.sum(approx, dim=1)
            #sq_err = -torch.sum(torch.square(approx))            
            sq_err = torch.sum(torch.square(curr_value - approx) - torch.square(approx))            
            
            if is_train: sq_err.backward()
            _loss += sq_err.item()
            
        return _loss

    
    '''
        cluster_label: num_rows
    '''
    def clustering(self, args):
        # Clustering
        cluster_label = torch.zeros(self.num_first_dim, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if args.is_dense:
                for i in tqdm(range(0, self.num_first_dim, args.batch_nz)):                
                    curr_batch_size = min(self.num_first_dim - i, args.batch_nz)
                    assert(curr_batch_size > 1)
                    #dist = torch.zeros((self.tensor.max_first, curr_batch_size, self.tensor.max_first), device=self.device)   # i_max x batch size x i_max
                    curr_U = self.U[i:i+curr_batch_size]  # bs' x rank                
                    curr_dist = curr_U.unsqueeze(0) - self.centroids.unsqueeze(1) # num_cents x bs' x rank
                    curr_dist = torch.sum(torch.square(curr_dist), dim=-1) # num_cents x bs'
                    #dist[j,:,:] = curr_dist

                    cluster_label[i:i+curr_batch_size] = torch.argmin(curr_dist, dim=0)  # bs'
            else:
                for i in range(0, self.tensor.num_tensor, args.batch):                
                    curr_batch_size = min(self.tensor.num_tensor - i, args.batch)
                    assert(curr_batch_size > 1)
                    #dist = torch.zeros((self.tensor.max_first, curr_batch_size, self.tensor.max_first), device=self.device)   # i_max x batch size x i_max
                    curr_U = self.U[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]  # bs' x rank                
                    curr_dist = curr_U.unsqueeze(0) - self.centroids.unsqueeze(1) # num_cents x bs' x rank
                    curr_dist = torch.sum(torch.square(curr_dist), dim=-1) # num_cents x bs'
                    #dist[j,:,:] = curr_dist

                    cluster_label[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]] = torch.argmin(curr_dist, dim=0)  # bs'
        return cluster_label
        
        
    def quantization(self, args):
        optimizer = torch.optim.Adam([self.U, self.S, self.centroids] + self.V, lr=args.lr)
        max_fitness = -100
        for _epoch in range(args.epoch):
            optimizer.zero_grad()
            # Clustering     
            self.mapping = self.clustering(args)  # num_rows, row idx of U -> idx of cluster
            if args.is_dense:
                self.shuffled_mapping = self.mapping[self.random_idx]
            
            if args.is_dense:
                self.L2_loss_dense(args, True, "train") 
            else:
                self.L2_loss(args, True, "train")
                
            # cluster loss
            if args.is_dense:
                for i in range(0, self.num_first_dim, args.batch_nz):                
                    curr_batch_size = min(args.batch_nz, self.num_first_dim - i)
                    assert(curr_batch_size > 1)
                    curr_mapping = self.mapping[i:i+curr_batch_size]   # bs'
                    curr_U = self.U[i:i+curr_batch_size] 
                    curr_U_cluster = self.centroids[curr_mapping]
                    cluster_loss = torch.sum(torch.square(curr_U_cluster - curr_U.detach()))            
                    cluster_loss.backward()
            else:
                for i in range(0, self.tensor.num_tensor, args.batch):                
                    curr_batch_size = min(args.batch, self.tensor.num_tensor - i)
                    assert(curr_batch_size > 1)
                    curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]   # bs'
                    curr_U = self.U[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]] 
                    curr_U_cluster = self.centroids[curr_mapping]
                    cluster_loss = torch.sum(torch.square(curr_U_cluster - curr_U.detach()))            
                    cluster_loss.backward()
            #del curr_mapping, curr_U, curr_U_cluster
            #clear_memory()
            
            optimizer.step()            
            if (_epoch + 1) % 10 == 0:
                with torch.no_grad():
                    self.mapping = self.clustering(args)
                    if args.is_dense:
                        self.shuffled_mapping = self.mapping[self.random_idx]
                        _loss = self.L2_loss_dense(args, False, "test") 
                    else:
                        _loss = self.L2_loss(args, False, "test")              
                    _fitness = 1 - math.sqrt(_loss)/math.sqrt(self.tensor.sq_sum)
                    print(f'epoch: {_epoch}, l2 loss: {_loss}, fitness: {_fitness}')
                    with open(args.output_path + ".txt", 'a') as f:
                        f.write(f'epoch: {_epoch}, l2 loss: {_loss}, fitness: {_fitness}\n')
                       
                    if _fitness > max_fitness:
                        max_fitness = _fitness
                        final_U = self.U.data.clone().detach().cpu()
                        final_cents = self.centroids.data.clone().detach().cpu()
                        final_V = [_v.data.clone().detach().cpu() for _v in self.V]                        
                        final_S = self.S.data.clone().detach().cpu()                        
                        final_mapping = self.mapping.clone().detach().cpu()
                        
        self.centroids.data.copy_(final_cents.to(self.device))
        self.U.data.copy_(final_U.to(self.device))
        
        for m in range(self.tensor.order-2):
            self.V[m].data.copy_(final_V[m].to(self.device))
        self.S.data.copy_(final_S.to(self.device))
        self.mapping = final_mapping.to(self.device)        
        
        torch.save({
            'fitness': max_fitness, 'centroids': self.centroids.data, 'mapping': self.mapping,
            'S': self.S.data, 'V': final_V,
        }, args.output_path + "_cp.pt")
                   
            
    '''
        Use cpd to initialize tucker decomposition
    '''
    def init_tucker(self, args):
        #self.mapping = self.clustering(args)  # k x i_max
        #self.mapping_mask = torch.zeros(self.tensor.num_tensor, self.tensor.max_first, dtype=torch.bool, device=self.device)   # k x i_max
        #for _k in range(self.tensor.num_tensor):        
        #    self.mapping_mask[_k, :self.tensor.first_dim[_k]] = True
        self.G = torch.zeros([self.rank]*self.tensor.order, device=self.device, dtype=torch.double)    
        self.G = self.G.fill_diagonal_(1)
    
    
    def L2_loss_tucker(self, batch_loss_zero, batch_loss_nz):             
        with torch.no_grad():  
            # Define mat_G
            perm_dims = tuple([m for m in range(1, self.tensor.order-1)] + [0, self.tensor.order-1])
            mat_G = torch.reshape(torch.permute(self.G, perm_dims), (-1, self.rank**2))   # R^(d-2) x R^2    

            # Define VtV
            Vkron = None           
            for m in range(self.tensor.order-2):
                if m == 0: 
                    Vkron = self.V[m].unsqueeze(0)
                else:
                    Vkron = batch_kron(Vkron, self.V[m].unsqueeze(0))                   
            Vkron = Vkron.squeeze()   # i_2*...*i_(m-1) x R^(d-2)
            
            # Defin VG
            VG = Vkron @ mat_G   # i_2*...*i_(m-1) x R^2  
            first_mat = 0
            
            for i in range(0, self.tensor.num_tensor, batch_loss_zero):                
                curr_batch_size = min(self.tensor.num_tensor - i, batch_loss_zero)
                assert(curr_batch_size > 1)                
                curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]   # batch size'
                curr_U = self.centroids.data[curr_mapping]   # batch size' x R
                
                final_idx = self.U_mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]
                curr_S = self.S[i:i+curr_batch_size, :].data   # batch size' x R
                UtU_input = torch.bmm(curr_U.unsqueeze(-1), curr_U.unsqueeze(1))   # batch size' x R x R                                                
                UtU = torch.zeros((curr_batch_size, self.rank, self.rank), device=self.device, dtype=torch.double)                
                UtU = UtU.index_add_(0, final_idx-i, UtU_input)
                StS = torch.bmm(curr_S.unsqueeze(-1), curr_S.unsqueeze(1))                            
                first_mat = first_mat + torch.sum(batch_kron(UtU, StS), dim=0)   # R^2 x R^2
           
            first_mat = VG @ first_mat  # i_2*...*i_(m-1) x R^2              
            sq_loss = torch.sum(first_mat * VG).item()          
            
            # Correct non-zero terms
            for i in tqdm(range(0, self.tensor.num_nnz, batch_loss_nz)):
                curr_batch_size = min(self.tensor.num_nnz - i, batch_loss_nz)                
                assert(curr_batch_size > 1)
                curr_vals = torch.tensor(self.tensor.values[i:i+curr_batch_size], dtype=torch.double, device=self.device)
                first_idx = torch.tensor(self.tensor.indices[0][i:i+curr_batch_size], dtype=torch.long, device=self.device) 
                last_idx = torch.tensor(self.tensor.indices[-1][i:i+curr_batch_size], dtype=torch.long, device=self.device)                         

                map_input = self.U_sidx[last_idx] + first_idx
                u_input = self.mapping[map_input]    
                first_mat = self.centroids.data[u_input]  # batch size x rank
                for m in range(self.tensor.order-2):      
                    curr_idx = torch.tensor(self.tensor.indices[m+1][i:i+curr_batch_size], dtype=torch.long, device=self.device)
                    first_mat = face_split(first_mat, self.V[m][curr_idx, :])  

                first_mat = face_split(first_mat, self.S[last_idx, :])   # batch size x rank...
                vec_G = torch.flatten(self.G).unsqueeze(-1)  # rank..... x 1 
                approx = torch.mm(first_mat, vec_G).squeeze()     
                curr_loss = torch.sum(torch.square(approx - curr_vals)) - torch.sum(torch.square(approx))              
                sq_loss = sq_loss + curr_loss.item()
                
        return sq_loss
    
    
    def L2_loss_tucker_dense(self, batch_size):
        _loss = 0
        for m in range(1, self.tensor.order-2):
            if m == 1:
                VKron = self.V[m].data.unsqueeze(0)  # 1 x j_2 x R
            else:
                VKron = batch_kron(VKron, self.V[m].data.unsqueeze(0))              
        if self.tensor.order > 3:
            VKron = VKron.squeeze()    
        # j_2...j_(d-2) x R^(d-3)
            
        with torch.no_grad():        
            for i in tqdm(range(0, self.num_first_dim, batch_size)):         
                curr_batch_size = min(batch_size, self.num_first_dim - i)
                assert(curr_batch_size > 1)
                curr_tensor = self.set_curr_tensor_new(curr_batch_size, i, 1)    # bs'*j_1 x j_2*...*j_(d-2)         
                curr_U = self.centroids[self.mapping[i:i+curr_batch_size]] # bs' x R         
                curr_UV = batch_kron(curr_U.unsqueeze(0), self.V[0].unsqueeze(0)).squeeze()   # bs'*j_1 x R^2
                
                curr_G = torch.reshape(self.G, (self.rank*self.rank, -1))    # R^2 x R^(d-2)                
                UVG = curr_UV @ curr_G   # bs'*j_1 x R^(d-2)
                UVG = torch.reshape(UVG, (curr_batch_size, self.V[0].shape[0], -1))  # bs' x j_1 x R^(d-2)
                                            
                VS = self.S[self.U_mapping[i:i+curr_batch_size]].unsqueeze(1)  # bs' x 1 x R          
                if self.tensor.order > 3:
                    VS = batch_kron(VKron.repeat(curr_batch_size, 1, 1), VS)   # bs' x j_2...j_(d-2) x R^(d-2)
                VS = torch.transpose(VS, 1, 2)   # bs' x R^(d-2) x j_2...j_(d-2) 
                approx = torch.bmm(UVG, VS)   # bs' x j_1 x j_2...j_(d-2)
                approx = torch.reshape(approx, (-1, approx.shape[-1]))
                curr_loss = torch.sum(torch.square(approx - curr_tensor))        
                _loss += curr_loss.item()

        return _loss
    
    
    def als_U(self, args):    
        with torch.no_grad():                   
            num_clusters = self.centroids.shape[0]
            first_mat = torch.zeros(num_clusters, self.rank, device=self.device, dtype=torch.double)   # i_max x rank
            second_mat = torch.zeros(num_clusters, self.rank, self.rank, device=self.device, dtype=torch.double)    # i_max x rank x rank
            mat_G = torch.reshape(self.G, (self.rank, -1))   # rank x rank^(d-1)
            
            # Define Vkron
            Vkron = None
            if args.is_dense:
                for m in range(1, self.tensor.order-2):
                    if m == 1:
                        Vkron = self.V[m].data.unsqueeze(0)  # 1 x j_1 x R
                    else:
                        Vkron = batch_kron(Vkron, self.V[m].data.unsqueeze(0))
                        
                if self.tensor.order > 3:
                    Vkron = Vkron.squeeze()   # j_2*...*j_(d-2) x R^(d-2)

            # Define VtV
            VtV = None
            for m in range(0, self.tensor.order-2):
                if m == 0:
                    VtV = torch.mm(self.V[m].data.t(), self.V[m].data).unsqueeze(0)  # 1 x R x R
                else:
                    curr_VtV = torch.mm(self.V[m].data.t(), self.V[m].data).unsqueeze(0)  # 1 x R x R
                    VtV = batch_kron(VtV, curr_VtV)  
            VtV = VtV.squeeze()  # R^(d-2) x R^(d-2)

            # first mat
            if args.is_dense:                
                for i in tqdm(range(0, self.num_first_dim, args.batch_nz)):                
                    curr_batch_size = min(args.batch_nz, self.num_first_dim - i)
                    assert(curr_batch_size > 1)                    
                    curr_tensor = self.set_curr_tensor_new(curr_batch_size, i, 0)  # bs'xj_1*j_2*...*j_(d-2)     
                    curr_tensor = torch.reshape(curr_tensor, (curr_batch_size, self.V[0].shape[0], -1))   # bs' x j_1 x j_2*...*j_(d-2)     
                    temp_V = self.V[0].repeat(curr_batch_size, 1, 1)   # bs' x j_1 x R
                    VX = torch.bmm(torch.transpose(temp_V, 1, 2), curr_tensor)   # bs' x R x j_2*...*j_(d-2)     
                    
                    curr_idx = self.U_mapping[i:i+curr_batch_size]  # bs'
                    VS = self.S[curr_idx, :].unsqueeze(1)   # bs' x R
                    if self.tensor.order > 3:
                        VS = batch_kron(Vkron.repeat(curr_batch_size, 1, 1), VS)  # bs' x j_2*...*j_(d-2) x R^(d-2)
                    VXVS = torch.bmm(VX, VS)  # bs' x R x R^(d-2)    
                    XVS = torch.reshape(VXVS, (curr_batch_size, -1)).unsqueeze(1)   # bs' X 1 X R^(d-1)                    
                    XVSG = torch.bmm(XVS, mat_G.t().repeat(curr_batch_size, 1, 1)).squeeze()   # bs' x R                                        
                    curr_mapping = self.mapping[i:i+curr_batch_size]  # bs'
                    first_mat = first_mat.index_add_(0, curr_mapping, XVSG)
            else:                                        
                for i in tqdm(range(0, self.tensor.num_nnz, args.batch_nz)):          
                    curr_batch_size = min(args.batch_nz, self.tensor.num_nnz - i)
                    assert(curr_batch_size > 1)
                    
                    last_idx = self.tensor.indices[-1][i:i+curr_batch_size]
                    last_idx = torch.tensor(last_idx, dtype=torch.long, device=self.device)
                    first_idx = self.tensor.indices[0][i:i+curr_batch_size]
                    first_idx = torch.tensor(first_idx, dtype=torch.long, device=self.device)                                  
                    temp_S = self.S[last_idx, :]   # bs'x rank
                
                    XVSG = None   # bs' x rank^(d-2)
                    for m in range(self.tensor.order-2):
                        curr_idx = self.tensor.indices[m+1][i:i+curr_batch_size]
                        curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)
                        curr_V = self.V[m][curr_idx, :]   # bs' x rank                        
                        if m == 0:
                            XVSG = curr_V  # bs' x rank
                        else:
                            XVSG = face_split(XVSG, curr_V) 
                                                            
                    XVSG = face_split(XVSG, temp_S)   # bs'x rank^(d-1)                    
                    curr_values = self.tensor.values[i:i+curr_batch_size]
                    curr_values = torch.tensor(curr_values, dtype=torch.double, device=self.device)  # bs'
                    XVSG = curr_values.unsqueeze(-1) * XVSG  # bs'x rank^(d-1)
                    
                    XVSG = XVSG @ mat_G.t()   # bs' x rank          
                    curr_idx = first_idx + self.U_sidx[last_idx]
                    temp_mapping = self.mapping[curr_idx]   # bs'
                    first_mat = first_mat.index_add_(0, temp_mapping, XVSG)   
                
            # Second mat
            for i in tqdm(range(0, self.tensor.num_tensor, args.batch)):                
                curr_batch_size = min(self.tensor.num_tensor - i, args.batch)
                assert(curr_batch_size > 1)                
                curr_idx = self.U_mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]] # bs'
                curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]] # bs'
                curr_S = self.S.data[curr_idx, :]   # bs' x rank
                StS = torch.bmm(curr_S.unsqueeze(-1), curr_S.unsqueeze(1))   # bs' x rank x rank                                     
                second_mat = second_mat.index_add_(0, curr_mapping, StS)   # i_max x rank x rank
                 
            for i in tqdm(range(num_clusters)):
                temp_sm = batch_kron(VtV.unsqueeze(0), second_mat[i].unsqueeze(0))   # R^(d-1) x R^(d-1)
                temp_sm = temp_sm.squeeze()
                temp_sm = mat_G @ temp_sm @ mat_G.t()  # R X R
                second_mat[i] = temp_sm
                 
            self.centroids.data = torch.bmm(first_mat.unsqueeze(1), torch.linalg.pinv(second_mat)).squeeze()                          
            
            
    def als_V(self, args, mode):            
        with torch.no_grad():
            # Define mat_G
            perm_dims = [mode, 0, self.tensor.order-1] + [m for m in range(1, self.tensor.order-1) if m != mode]
            mat_G = torch.reshape(torch.permute(self.G, perm_dims), (self.rank, -1))   # R x R^(d-1)                
            # define VtV
            VtV, Vkron = None, None
            if self.tensor.order > 3:
                _cnt = 0
                for m in range(1, self.tensor.order-1): 
                    if m == mode: continue
                    if _cnt == 0:
                        VtV = (self.V[m-1].t()@self.V[m-1]).unsqueeze(0)
                    else:
                        VtV = batch_kron(VtV, (self.V[m-1].t()@self.V[m-1]).unsqueeze(0))            
                    _cnt += 1
                VtV = VtV.squeeze()  
                # rank^(d-3) x rank^(d-3)
                                
                if args.is_dense:
                    _cnt = 0
                    for m in range(1, self.tensor.order-1):
                        if m == mode: continue
                        if _cnt == 0:
                            Vkron = self.V[m-1].unsqueeze(0)  # 1 x i_m x R
                        else:
                            Vkron = batch_kron(Vkron, self.V[m-1].unsqueeze(0))
                        _cnt += 1
                    Vkron = Vkron.squeeze()   # i_2*...*i_(d-1) x R^(d-3)
         
            # first_mat
            first_mat = torch.zeros((self.tensor.middle_dim[mode-1], self.rank), dtype=torch.double, device=self.device) # i_m x R            
            if args.is_dense:
                for i in tqdm(range(0, self.num_first_dim, args.batch_nz)):                    
                    # Build the first mat
                    curr_batch_size = min(args.batch_nz, self.num_first_dim - i)        
                    assert(curr_batch_size > 1)
                    curr_mapping = self.mapping[i:i+curr_batch_size]   # bs'
                    curr_U = self.centroids.data[curr_mapping]  # bs' x rank
                    curr_U_mapping = self.U_mapping[i:i+curr_batch_size]  # bs'
                    curr_S = self.S[curr_U_mapping]  # bs' x R
                        
                    curr_tensor = self.tensor.src_tensor_torch[i:i+curr_batch_size].to(self.device)  # bs' x j_1 x j_2 x ... x j_(m-2)           
                    perm_dims = [mode, 0] + [m for m in range(1, self.tensor.order-1) if m != mode]
                    curr_tensor = torch.permute(curr_tensor, perm_dims)
                    # j_(mode) x bs' x j_1 x ... x j_(m-2)
                    curr_tensor = torch.reshape(curr_tensor, (curr_tensor.shape[0], -1))   
                    # j_mode x bs'*j_1* ..* j_(m-2)                                                       
                    USV = face_split(curr_U, curr_S)   # bs' x R^2                    
                    if self.tensor.order > 3:
                        USV = batch_kron(USV.unsqueeze(0), Vkron.unsqueeze(0)).squeeze()  # bs'*j_1*...*j_(m-2) x R^(d-1)
                        
                    temp_fm = curr_tensor @ USV   # j_(mode) x R^(d-1)
                    temp_fm = temp_fm @ mat_G.t()   #  j_(mode) x R                    
                    first_mat = first_mat + temp_fm
            else:
                for i in tqdm(range(0, self.tensor.num_nnz, args.batch_nz)):
                    curr_batch_size = min(args.batch_nz, self.tensor.num_nnz - i) 
                    assert(curr_batch_size > 1)
                        
                    first_idx = self.tensor.indices[0][i:i+curr_batch_size]   # bs
                    first_idx = torch.tensor(first_idx, dtype=torch.long, device=self.device)  # bs
                    last_idx = self.tensor.indices[-1][i:i+curr_batch_size]   # bs
                    last_idx = torch.tensor(last_idx, dtype=torch.long, device=self.device)  # bs
                    
                    curr_mapping = self.mapping[first_idx + self.U_sidx[last_idx]]   # bs
                    curr_U = self.centroids.data[curr_mapping]  # bs x R                    
                    curr_S = self.S[last_idx]  # bs x R
                    
                    USV = face_split(curr_U, curr_S)  # bs x R^2
                    for m in range(1, self.tensor.order - 1):
                        if m == mode: continue
                        else:
                            curr_idx = self.tensor.indices[m][i:i+curr_batch_size]        
                            curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)  # bs'                            
                            USV = face_split(USV, self.V[m-1][curr_idx, :])                                                                 # XUSV: bs' x R^(d-1)

                    curr_values = self.tensor.values[i:i+curr_batch_size]   # bs'
                    curr_values = torch.tensor(curr_values, dtype=torch.double, device=self.device)  # bs'
                    XUSV = curr_values.unsqueeze(-1) * USV  # bs' x R^(d-1)
                    temp_first_mat = XUSV @ mat_G.t()   # bs' x R

                    curr_idx = self.tensor.indices[mode][i:i+curr_batch_size]  # bs'
                    curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)  # bs'
                    first_mat = first_mat.index_add_(0, curr_idx, temp_first_mat)  # j_m x R
                    
            # second mat
            second_mat = 0
            for i in tqdm(range(0, self.tensor.num_tensor, args.batch)):  
                curr_batch_size = min(self.tensor.num_tensor - i, args.batch)
                assert(curr_batch_size > 1)
                curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]   # bs'
                curr_U = self.centroids.data[curr_mapping]  # bs' x rank
                U_input = torch.bmm(curr_U.unsqueeze(-1), curr_U.unsqueeze(1))  # bs' x rank x rank
                UtU = torch.zeros((curr_batch_size, self.rank, self.rank), device=self.device, dtype=torch.double)
                curr_U_mapping = self.U_mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]  # bs'
                UtU = UtU.index_add_(0, curr_U_mapping - i, U_input)    # bs x rank x rank
                
                curr_S = self.S[i:i+curr_batch_size, :]  # bs x R                
                StS = torch.bmm(curr_S.unsqueeze(-1), curr_S.unsqueeze(1)) # bs x R x R
                second_mat = second_mat + torch.sum(batch_kron(UtU, StS), dim=0)   # R^2 x R^2
                        
            if self.tensor.order > 3:
                second_mat = batch_kron(second_mat.unsqueeze(0), VtV.unsqueeze(0))
                second_mat = second_mat.squeeze()
            # R^(d-1) x R^(d-1)
             
            second_mat = mat_G @ second_mat @ mat_G.t()   # R X R
            second_mat = torch.linalg.pinv(second_mat)
            self.V[mode-1].data = first_mat @ second_mat

            
    def als_S(self, args):        
        with torch.no_grad():
            # Define mat_G
            perm_dims = [self.tensor.order-1] + [m for m in range(self.tensor.order-1)]
            mat_G = torch.reshape(torch.permute(self.G, perm_dims), (self.rank, -1))   # R x R^(d-1)    
            
            # Define VtV                        
            VtV = None
            for m in range(0, self.tensor.order-2):
                if m == 0:
                    VtV = torch.mm(self.V[m].data.t(), self.V[m].data).unsqueeze(0)  # 1 x R x R
                else:
                    curr_VtV = torch.mm(self.V[m].data.t(), self.V[m].data).unsqueeze(0)  # 1 x R x R
                    VtV = batch_kron(VtV, curr_VtV)  
            VtV = VtV.squeeze()  # R^(d-2) x R^(d-2)

            # Define Vkron
            Vkron = None
            if args.is_dense:
                for m in range(0, self.tensor.order-2):
                    if m == 0:
                        Vkron = self.V[m].unsqueeze(0)
                    else:
                        Vkron = batch_kron(Vkron, self.V[m].unsqueeze(0))

                Vkron = Vkron.squeeze()  # i_2*...i_m-1 x R^(d-2)                
                        
            first_mat = torch.zeros((self.tensor.num_tensor, self.rank), dtype=torch.double, device=self.device)
            if args.is_dense:
                for i in tqdm(range(0, self.num_first_dim, args.batch_nz)):                                  
                    curr_batch_size = min(args.batch_nz, self.num_first_dim - i)
                    assert(curr_batch_size > 1)  
                    curr_mapping = self.mapping[i:i+curr_batch_size]   # bs'
                    curr_U = self.centroids.data[curr_mapping]   # bs' x R                    
                    curr_tensor = self.set_curr_tensor_new(curr_batch_size, i, 0)  # bs' x i_2*...*i_(m-1)                                                              
                    UX = torch.bmm(curr_U.unsqueeze(-1), curr_tensor.unsqueeze(1)) # bs' x R x i_2*...*i_(m-1)                                                              
                    #print(UX.shape)
                    #print(Vkron.repeat(curr_batch_size, 1, 1).shape)
                    UXV = torch.bmm(UX, Vkron.repeat(curr_batch_size, 1, 1))  # bs' x R x R^(d-2)
                    temp_fm = torch.reshape(UXV, (curr_batch_size, -1))  # bs' x R^(d-1)                    
                    temp_fm = temp_fm @ mat_G.t()  # bs' x R
                    
                    curr_U_mapping = self.U_mapping[i:i+curr_batch_size]  # bs'
                    first_mat = first_mat.index_add_(0, curr_U_mapping, temp_fm)
                                        
            else:
                for i in tqdm(range(0, self.tensor.num_nnz, args.batch_nz)):    
                    curr_batch_size = min(args.batch_nz, self.tensor.num_nnz - i)
                    assert(curr_batch_size > 1)
                    
                    first_idx = self.tensor.indices[0][i:i+curr_batch_size]   # bs
                    first_idx = torch.tensor(first_idx, dtype=torch.long, device=self.device)  # bs
                    last_idx = self.tensor.indices[-1][i:i+curr_batch_size]   # bs
                    last_idx = torch.tensor(last_idx, dtype=torch.long, device=self.device)  # bs

                    curr_mapping = self.mapping[first_idx + self.U_sidx[last_idx]]   # bs
                    XUVG = self.centroids.data[curr_mapping]  # bs x R                    
                    for m in range(1, self.tensor.order-1):
                        curr_idx = self.tensor.indices[m][i:i+curr_batch_size]
                        curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)
                        curr_V = self.V[m-1].data[curr_idx, :]   # bs' x rank                        
                        XUVG = face_split(XUVG, curr_V)
                    # bs' x R^(d-1)
                    
                    curr_values = self.tensor.values[i:i+curr_batch_size]   # bs
                    curr_values = torch.tensor(curr_values, dtype=torch.double, device=self.device)  # bs
                    XUVG = curr_values.unsqueeze(-1) * XUVG   # bs x R^(d-1)
                    XUVG = XUVG @ mat_G.t()   # bs x R              
                    first_mat = first_mat.index_add_(0, last_idx, XUVG)  # num_tensor x rank
            
            # second matrix            
            UtU = torch.bmm(self.centroids.data.unsqueeze(-1), self.centroids.data.unsqueeze(1)) 
            # num_cents x R x R
            num_clusters = self.centroids.data.shape[0]
            GUVG = torch.zeros((num_clusters, self.rank, self.rank), dtype=torch.double, device=self.device)
            # num_cents x R x R
            
            for i in tqdm(range(num_clusters)):              
                UV = batch_kron(UtU[i].unsqueeze(0), VtV.unsqueeze(0)).squeeze()  # R^(d-1) x R^(d-1)
                GUV = mat_G @ UV    # R x R^(d-1)
                GUVG[i] = GUV @ mat_G.t()  # R x R
            # num_cents x R x R
            del UV, GUV
            
            second_mat = torch.zeros((self.tensor.num_tensor, self.rank, self.rank), device=self.device, dtype=torch.double)
            for i in tqdm(range(0, self.tensor.num_tensor, args.batch)):                       
                curr_batch_size = min(args.batch, self.tensor.num_tensor - i)
                curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]  # bs'
                
                temp_sm = torch.zeros((curr_batch_size, self.rank, self.rank), device=self.device, dtype=torch.double)
                
                curr_U_mapping = self.U_mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]  # bs'
                second_mat[i:i+curr_batch_size] = temp_sm.index_add_(0, curr_U_mapping-i, GUVG[curr_mapping])                
            
            second_mat = torch.linalg.pinv(second_mat)     # batch size x R x R                
            self.S.data = torch.bmm(first_mat.unsqueeze(1), second_mat).squeeze()
                
                
    def als_G(self, args):
        with torch.no_grad():                     
            # define VtV
            VtV = None
            Vkron = None
            for m in range(self.tensor.order-2):
                if m == 0: 
                    VtV = torch.mm(self.V[m].t(), self.V[m]).unsqueeze(0)
                    if args.is_dense:
                        Vkron = self.V[m].data.unsqueeze(0)
                else:
                    VtV = batch_kron(VtV, torch.mm(self.V[m].t(), self.V[m]).unsqueeze(0))                         
                    if args.is_dense:
                        Vkron = batch_kron(Vkron, self.V[m].data.unsqueeze(0))
                
            VtV = VtV.squeeze()   # rank x rank        
            if args.is_dense:
                Vkron = Vkron.squeeze()  # i_2*...i_m-1 x R^(d-2)
            
            # first mat
            first_mat = torch.linalg.pinv(VtV)            
            # Second mat
            second_mat = 0
            if args.is_dense:
                for i in tqdm(range(0, self.num_first_dim, args.batch_nz)):                
                    curr_batch_size = min(args.batch_nz, self.num_first_dim - i)               
                    assert(curr_batch_size > 1)
                    curr_mapping = self.mapping[i:i+curr_batch_size]   # bs'
                    curr_U = self.centroids[curr_mapping].data # bs' x R    
                    curr_idx = self.U_mapping[i:i+curr_batch_size]
                    
                    curr_S = self.S.data[curr_idx, :]   # bs' x rank                    
                    US = face_split(curr_U, curr_S)    # bs' x rank^2                    
                    
                    # Build tensor
                    curr_X = self.tensor.src_tensor_torch[i:i+curr_batch_size].to(self.device)  # bs' x i_2 x ... x i_(m-1)                                          
                    curr_X = torch.reshape(curr_X, (curr_batch_size, -1))  # bs' x i_2...i_(m-1)                                         
                                        
                    XUS = curr_X.unsqueeze(-1) * US.unsqueeze(1) # bs' x i_2...i_(m-1) x rank^2               
                    second_mat = second_mat + torch.sum(XUS, dim=0)   # i_2...i_(m-1) x rank^2                    
                second_mat = Vkron.t() @ second_mat  #  R^(d-2) x R^2
            else:    
                for i in tqdm(range(0, self.tensor.num_nnz, args.batch_nz)):                
                    curr_batch_size = min(args.batch_nz, self.tensor.num_nnz - i)                                                       
                    curr_fi = self.tensor.indices[0][i:i+curr_batch_size]
                    curr_li = self.tensor.indices[-1][i:i+curr_batch_size]
                    curr_fi = torch.tensor(curr_fi, dtype=torch.long, device=self.device)
                    curr_li = torch.tensor(curr_li, dtype=torch.long, device=self.device)                    
                    map_input = curr_fi + self.U_sidx[curr_li]  # bs'
                    curr_mapping = self.mapping[map_input]   # bs'
                    curr_U = self.centroids[curr_mapping].data   # k dist x R                
                    
                    VX = None
                    for m in range(self.tensor.order - 2):
                        curr_idx = self.tensor.indices[m+1][i:i+curr_batch_size]
                        curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)
                        if m == 0:                        
                            VX = self.V[m][curr_idx, :].data   # bs' x rank
                        else:
                            VX = face_split(VX, self.V[m][curr_idx, :].data)   
                    curr_values = torch.tensor(self.tensor.values[i:i+curr_batch_size], dtype=torch.float, device=self.device)   # bs'
                    VX = VX * curr_values.unsqueeze(-1)   # VXUS: bs' x R^(d-2)                   
                    US = face_split(curr_U, self.S[curr_li, :].data)  # bs' x rank^2                                                            
                    temp_sm = VX.unsqueeze(-1) * US.unsqueeze(1)
                    second_mat = second_mat + torch.sum(temp_sm, dim=0)  # R^(d-2) x R^2
                                    
            # Third mat
            US = 0
            for i in tqdm(range(0, self.tensor.num_tensor, args.batch)):
                curr_batch_size = min(args.batch, self.tensor.num_tensor - i)                
                assert(curr_batch_size > 1)
                curr_mapping = self.mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]   # bs'
                curr_U = self.centroids[curr_mapping].data    # bs' x R       
                curr_S = self.S.data[i:i+curr_batch_size, :]    # batch size x rank
                UtU_input = torch.bmm(curr_U.unsqueeze(-1), curr_U.unsqueeze(1))    # bs' x rank x rank
                UtU = torch.zeros((curr_batch_size, self.rank, self.rank), device=self.device, dtype=torch.double)
                UtU_idx = self.U_mapping[self.U_sidx[i]:self.U_sidx[i+curr_batch_size]]
                UtU = UtU.index_add_(0, UtU_idx - i, UtU_input)
                
                StS = torch.bmm(curr_S.unsqueeze(-1), curr_S.unsqueeze(1))   # batch size x rank x rank
                temp_US = batch_kron(UtU, StS)   # batch size x rank^2 x rank^2
                US = US + torch.sum(temp_US, dim=0)            
            third_mat = torch.linalg.pinv(US)
                                                 
            self.G = first_mat @ second_mat @ third_mat
            orig_shape = tuple(self.rank for _ in range(self.tensor.order))
            self.G = torch.reshape(self.G, orig_shape)
            
            temp_perm = tuple([self.tensor.order-2] + [m for m in range(self.tensor.order-2)] + [self.tensor.order-1])
            self.G = torch.permute(self.G, temp_perm)
            
            
    def als(self, args):                
        self.init_tucker(args) 
        if args.is_dense:
            sq_loss = self.L2_loss_tucker_dense(args.batch_nz)
        else:
            sq_loss = self.L2_loss_tucker(args.batch, args.batch_nz)
        prev_fit = 1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)        
        print(f'loss after loaded:{prev_fit}')        
        clear_memory()        
        
        prev_fit = 0
        for e in range(args.epoch_als):              
            self.als_G(args)                                                   
            self.als_U(args)            
            clear_memory()                        
                        
            for m in range(1, self.tensor.order-1):
                self.als_G(args)               
                self.als_V(args, m)                              
                clear_memory()            
                
            self.als_G(args)            
            self.als_S(args)                        
            clear_memory()            
            
            if args.is_dense:
                sq_loss = self.L2_loss_tucker_dense(args.batch_nz)
            else:
                sq_loss = self.L2_loss_tucker(args.batch, args.batch_nz)
            curr_fit = 1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)            
            with open(args.output_path + ".txt", 'a') as f:
                f.write(f'als epoch: {e+1}, after s:{curr_fit}\n')            
            print(f'als epoch: {e+1}, after s:{curr_fit}')                                    
            
            if curr_fit - prev_fit < 1e-4:                 
                break
            prev_fit = curr_fit                                      
            
        final_V = [_v.data.clone().detach().cpu() for _v in self.V]      
        torch.save({
            'fitness': curr_fit, 'mapping': self.mapping,
            'centroids': self.centroids.data, 'S': self.S.data, 'V': final_V, 'G': self.G}, args.output_path + ".pt")        
        