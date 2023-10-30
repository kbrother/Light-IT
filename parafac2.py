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
                self.U[self.U_sidx[i]:self.U_sidx[i+1],:] = torch.mm(Z, Ph) # i_max x rank

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
            
        UtU_input = torch.bmm(self.U.unsqueeze(-1), self.U.unsqueeze(1))   # i_sum x rank x rank                
        UtU = torch.zeros((self.tensor.num_tensor, self.rank, self.rank), device=self.device, dtype=torch.double)
        UtU = UtU.index_add_(0, self.U_mapping, UtU_input)   # k x rank x rank
        
        HtH = _H.t() @ _H
        #iden = torch.diag(torch.ones(self.rank)).to(self.device)
        print(torch.mean(torch.abs(UtU - HtH.unsqueeze(0))))        
            

    def __init__(self, _tensor, device, require_init, args):      
        # Intialization
        self.device = device
        self.tensor = _tensor
        if args.is_dense:
            scale_factor = 0.1
        else:
            scale_factor = 0.01
        
        _sum = 0
        self.U_sidx = [0]  # num_tensor + 1
        self.U_mapping = []  # i_sum
        for i in range(_tensor.num_tensor):
            _sum += _tensor.first_dim[i]
            self.U_sidx.append(_sum)
            for j in range(_tensor.first_dim[i]):
                self.U_mapping.append(i)
        
        self.U_mapping = torch.tensor(self.U_mapping, device=self.device, dtype=torch.int)
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
        
        with torch.no_grad():
            if args.is_dense:
                sq_loss = self.L2_loss_dense(args, False, "parafac2")
            else:
                sq_loss = self.L2_loss(args, False, "parafac2")
            print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)}') 
            print(f'square loss: {sq_loss}')
            
            
    '''
        input_U: num_tensor x i_max x rank
    '''
    def L2_loss_dense(self, args, is_train, mode):
        _loss = 0            
        for i in tqdm(range(0, self.tensor.num_tensor, args.batch_lossz)):                        
            Vprod = self.V[0]
            for j in range(1, self.tensor.order-2):
                Vprod = khatri_rao(Vprod, self.V[j]).unsqueeze(0)   # 1 x i_2 * ... * i_(m-1) x rank        

            
            curr_batch_size = min(args.batch_lossz, self.tensor.num_tensor - i) 
            assert(curr_batch_size > 1)
            curr_S = self.S[i:i+curr_batch_size, :].unsqueeze(1)  # bs x 1 x rank
            VS = Vprod * curr_S   # bs x i_2 * ... * i_(m-1) x rank        
            VS = torch.transpose(VS, 1, 2)   # bs x rank x i_2 * ... * i_(m-1)        
            
            curr_U = self.U[i:i+curr_batch_size]
            if mode != "parafac2":
                curr_mapping = self.mapping[i:i+curr_batch_size]
                curr_U_cluster = self.centroids[curr_mapping]
                if mode=="train":                
                    sg_part = (curr_U - curr_U_cluster).detach()
                    curr_U = curr_U - sg_part
                else:
                    curr_U = curr_U_cluster
             # curr_U: batch size x i_max x rank
            approx = torch.bmm(curr_U, VS)   # bs x i_max x i_2 * ... * i_(m-1)                    
            curr_tensor = self.set_curr_tensor(curr_batch_size, i)
            self.tensor.src_tensor_torch[i:i+curr_batch_size].to(self.device)  # bs x i_max x i_2 x ... x i_(m-1)  
            curr_tensor = torch.reshape(curr_tensor, (curr_batch_size, self.tensor.max_first, -1))   # bs x i_max x i_2 * ... * i_(m-1)  
            
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
        for i in tqdm(range(0, self.tensor.num_tensor, args.batch_lossz)):
            # zero terms             
            VtV = torch.ones((self.rank, self.rank), device=self.device, dtype=torch.double)  # r x r
            for j in range(self.tensor.order-2):
                VtV = VtV * (self.V[j].t() @ self.V[j])  # r x r
            
            curr_batch_size = min(args.batch_lossz, self.tensor.num_tensor - i)
            assert(curr_batch_size > 1)            
            
            curr_U = self.U[i:i+curr_batch_size] * self.U_mask[i:i+curr_batch_size]
            if mode != "parafac2":
                curr_mapping = self.mapping[i:i+curr_batch_size]
                curr_U_cluster = self.centroids[curr_mapping] * self.U_mask[i:i+curr_batch_size]
                if mode=="train":                
                    sg_part = (curr_U - curr_U_cluster).detach()
                    curr_U = curr_U - sg_part
                else:
                    curr_U = curr_U_cluster

            UtU = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U) # k x rank x rank
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
            
            mink = torch.min(final_idx)
            maxk = torch.max(final_idx) + 1                        
            curr_U = self.U[mink:maxk]
            if mode != "parafac2":
                curr_mapping = self.mapping[mink:maxk]
                if mode == "train":                
                    sg_part = (curr_U - self.centroids[curr_mapping]).detach()
                    curr_U = curr_U - sg_part
                else:
                    curr_U = self.centroids[curr_mapping]

            approx = curr_U[final_idx - mink, first_idx, :] * self.S[final_idx, :]  # bs x rank
            for m in range(1, self.tensor.order-1):
                curr_idx = torch.tensor(self.tensor.indices[m][i: i+curr_batch_size], device=self.device, dtype=torch.long)
                approx = approx * self.V[m-1][curr_idx, :]   # bs x rank
            
            curr_value = torch.tensor(self.tensor.values[i: i+curr_batch_size], dtype=torch.double, device=self.device)
            approx = torch.sum(approx, dim=1)
            sq_err = torch.sum(torch.square(curr_value - approx) - torch.square(approx))
            
            if is_train: sq_err.backward()
            _loss += sq_err.item()
            
        return _loss

    
    '''
        cluster_label: k x i_max
    '''
    def clustering(self, args):
        # Clustering
        cluster_label = torch.zeros(self.tensor.num_tensor, self.tensor.max_first, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for i in range(0, self.tensor.num_tensor, args.cluster_batch):                
                curr_batch_size = min(self.tensor.num_tensor - i, args.cluster_batch)
                assert(curr_batch_size > 1)
                #dist = torch.zeros((self.tensor.max_first, curr_batch_size, self.tensor.max_first), device=self.device)   # i_max x batch size x i_max
                curr_U = self.U[i:i+curr_batch_size, :, :]    # batch size x i_max x rank                
                curr_dist = curr_U.unsqueeze(0) - self.centroids.unsqueeze(1).unsqueeze(1) # i_max x batch size x i_max x rank
                curr_dist = torch.sum(torch.square(curr_dist), dim=-1) # i_max x batch size x i_max
                #dist[j,:,:] = curr_dist
                    
                cluster_label[i:i+curr_batch_size, :] = torch.argmin(curr_dist, dim=0)  # batch size x i_max
        return cluster_label
        
        
    def quantization(self, args):
        optimizer = torch.optim.Adam([self.U, self.S, self.centroids] + self.V, lr=args.lr)
        max_fitness = -100
        for _epoch in range(args.epoch):
            optimizer.zero_grad()
            # Clustering     
            self.mapping = self.clustering(args)
            
            if args.is_dense:
                self.L2_loss_dense(args, True, "train") 
            else:
                self.L2_loss(args, True, "train")
                
            # cluster loss
            for i in range(0, self.tensor.num_tensor, args.batch_lossz):
                curr_batch_size = min(args.batch_lossz, self.tensor.num_tensor - i)
                assert(curr_batch_size > 1)
                curr_mapping = self.mapping[i:i+curr_batch_size]
                curr_U = self.U[i:i+curr_batch_size] * self.U_mask[i:i+curr_batch_size]
                curr_U_cluster = self.centroids[curr_mapping] * self.U_mask[i:i+curr_batch_size]
                cluster_loss = torch.sum(torch.square(curr_U_cluster - curr_U.detach()))            
                cluster_loss.backward()
            del curr_mapping, curr_U, curr_U_cluster
            clear_memory()
            
            optimizer.step()            
            if (_epoch + 1) % 10 == 0:
                with torch.no_grad():
                    self.mapping = self.clustering(args)
                    if args.is_dense:
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
        self.mapping_mask = torch.zeros(self.tensor.num_tensor, self.tensor.max_first, dtype=torch.bool, device=self.device)   # k x i_max
        for _k in range(self.tensor.num_tensor):        
            self.mapping_mask[_k, :self.tensor.first_dim[_k]] = True
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
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
                curr_S = self.S[i:i+curr_batch_size, :].data   # batch size x R
                UtU = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U)   # batch size x R x R                
                StS = torch.bmm(curr_S.unsqueeze(-1), curr_S.unsqueeze(1))   # batch size x R x R
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
                
                min_k = torch.min(last_idx).item()
                max_k = torch.max(last_idx).item()
                # Prepare matrices
                curr_mapping = self.mapping[min_k:max_k+1, :]   # k_dist x i_max
                curr_U = self.centroids.data[curr_mapping] * self.U_mask[min_k:max_k+1]   # k_idst x i_max x R
                            
                first_mat = curr_U[last_idx - min_k, first_idx, :]  # batch size x rank
                for m in range(self.tensor.order-2):      
                    curr_idx = torch.tensor(self.tensor.indices[m+1][i:i+curr_batch_size], dtype=torch.long, device=self.device)
                    first_mat = face_split(first_mat, self.V[m][curr_idx, :])  

                first_mat = face_split(first_mat, self.S[last_idx, :])   # batch size x rank...
                vec_G = torch.flatten(self.G).unsqueeze(-1)  # rank..... x 1 
                approx = torch.mm(first_mat, vec_G).squeeze()     
                curr_loss = torch.sum(torch.square(approx - curr_vals)) - torch.sum(torch.square(approx))              
                sq_loss = sq_loss + curr_loss.item()
                
        return sq_loss
    
    
    '''
        Return a tensor of size 'batch size x i_max x j_1*j_2*...*j_(d-2)        
    '''
    def set_curr_tensor(self, batch_size, start_k):
        curr_tensor = self.tensor.src_tensor_torch[start_k:start_k+batch_size].to(self.device)  # bs x i_max x i_2 x ... x i_(m-1)  
        curr_tensor = torch.reshape(curr_tensor, (batch_size, self.tensor.max_first, -1))   # bs x i_max x i_2 * ... * i_(m-1) 
        return curr_tensor
        
                
    def L2_loss_tucker_dense(self, batch_size):
        _loss = 0
        print(self.tensor.order)
        for m in range(self.tensor.order-2):
            if m == 0:
                VKron = self.V[m].data.unsqueeze(0)  # 1 x j_1 x R
            else:
                VKron = batch_kron(VKron, self.V[m].data.unsqueeze(0))              
        VKron = VKron.squeeze()    
        # j_1...j_(d-2) x R^(d-2)
            
        with torch.no_grad():        
            for i in tqdm(range(0, self.tensor.num_tensor, batch_size)):         
                curr_batch_size = min(batch_size, self.tensor.num_tensor - i)
                assert(curr_batch_size > 1)
                curr_tensor = self.set_curr_tensor(curr_batch_size, i)    # batch size x i_max x J   
                curr_U = self.centroids[self.mapping[i:i+curr_batch_size, :]] * self.U_mask[i:i+curr_batch_size, :, :] # batch size x i_max x R               
                curr_G = torch.reshape(self.G, (self.rank, -1))    # R x R^(d-1)
                curr_G = curr_G.unsqueeze(0).repeat(curr_batch_size, 1, 1)  # batch size x R x R^(d-1)
                UG = torch.bmm(curr_U, curr_G)   # batch size x i_max x R^(d-1)
                                            
                curr_S = self.S[i:i+curr_batch_size, :].unsqueeze(1)  # batch size x 1 x R                               
                VS = batch_kron(VKron.repeat(curr_batch_size, 1, 1), curr_S)   # batch size x j_1...j_(d-2) x R^(d-1)
                VS = torch.transpose(VS, 1, 2)   # batch size x R^(d-1) x j_1...j_(d-2)
                approx = torch.bmm(UG, VS)   # batch size x i_max x j_1...j_(d-2)                
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
                for m in range(0, self.tensor.order-2):
                    if m == 0:
                        Vkron = self.V[m].data.unsqueeze(0)  # 1 x j_1 x R
                    else:
                        Vkron = batch_kron(Vkron, self.V[m].data.unsqueeze(0))
                Vkron = Vkron.squeeze()   # j_1*...*j_(d-2) x R^(d-2)

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
                for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_lossnz)):                
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_tensor - i)
                    assert(curr_batch_size > 1)
                    curr_mapping = self.mapping[i:i+curr_batch_size, :]  # bs x i_max
                    curr_mapping_mask = self.mapping_mask[i:i+curr_batch_size, :]  # bs x i_max
                    curr_S = self.S[i:i+curr_batch_size, :]   # bs x R

                    XVSG = batch_kron(Vkron.repeat(curr_batch_size, 1, 1), curr_S.unsqueeze(1))  # bs x j_1*...*j_(d-2) x R                 ^(d-1)
                    XVSG = torch.bmm(XVSG, mat_G.t().repeat(curr_batch_size, 1, 1))   # bs x j_1*...*j_(d-2) x R
                    curr_tensor = self.set_curr_tensor(curr_batch_size, i)  # batch size x i_max x j_1*j_2*...*j_(d-2)      
                    XVSG = torch.bmm(curr_tensor, XVSG)   # bs x i_max x R
                    XVSG = torch.reshape(XVSG, (curr_batch_size*self.tensor.max_first, -1))  # bs*i_max x R                   
                    first_mat = first_mat.index_add_(0, curr_mapping.flatten(), XVSG)
            else:                                        
                for i in tqdm(range(0, self.tensor.num_nnz, args.tucker_batch_lossnz)):          
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_nnz - i)
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
                    temp_mapping = self.mapping[last_idx, first_idx]   # bs'
                    first_mat = first_mat.index_add_(0, temp_mapping, XVSG)   
                
            # Second mat
            for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_alsnx)):                
                curr_batch_size = min(self.tensor.num_tensor - i, args.tucker_batch_alsnx)
                assert(curr_batch_size > 1)
                curr_S = self.S.data[i:i+curr_batch_size, :]   # bs x rank
                StS = torch.bmm(curr_S.unsqueeze(-1), curr_S.unsqueeze(1))   # bs x rank x rank
                StS = StS.unsqueeze(1).expand(-1, num_clusters, -1, -1)   # bs x num_cluster x rank x rank   
                
                curr_mapping = self.mapping[i:i+curr_batch_size, :]  # bs x num_cluster
                curr_mapping_mask = self.mapping_mask[i:i+curr_batch_size, :]  # bs x num_cluster
                curr_mapping = curr_mapping[curr_mapping_mask]
                StS = StS[curr_mapping_mask]
                
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
                            Vkron = self.V[m-1].unsqueeze()  # 1 x i_m x R
                        else:
                            Vkron = batch_kron(Vkron, self.V[m-1])
                        _cnt += 1
                    Vkron = Vkron.squeeze()   # i_2*...*i_(d-1) x R^(d-3)
         
            # first_mat
            first_mat = torch.zeros((self.tensor.middle_dim[mode-1], self.rank), dtype=torch.double, device=self.device) # i_m x R            
            if args.is_dense:
                for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_lossnz)):                    
                    # Build the first mat
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_tensor - i)        
                    assert(curr_batch_size > 1)
                    curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                    curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
                    curr_S = self.S[i:i+curr_batch_size, :]  # bs x R

                    curr_tensor = self.tensor.src_tensor_torch[i:i+curr_batch_size].to(self.device)  # bs x i_1 x i_2 x ... x i_(m-1)                 
                    perm_dims = [0, mode+1, 1] + [m for m in range(2, self.tensor.order-1) if m != mode + 1]    
                    curr_tensor = torch.permute(curr_tensor, perm_dims)   # bs x i_mode x i_1 x i_2 x ... x i_(m-1)  
                    curr_tensor = torch.reshape(curr_tensor, (curr_batch_size, self.tensor.middle_dim[mode-1], -1))   # bs x i_mode x i_1* ..* i_(m-1)              
                     
                    USV = curr_U  # batch size x i_max x R                
                    USV = batch_kron(USV, curr_S.unsqueeze(1))   # batch size x i_1 x R^2                    
                    if self.tensor.order > 3:
                        USV = batch_kron(USV, Vkron.repeat(curr_batch_size, 1, 1))   # batch size x i_1*...*i_(m-1) x R^(d-1)
                    
                    temp_fm = torch.bmm(curr_tensor, USV)   # batch size x i_m x R^(d-1)
                    temp_fm = torch.bmm(temp_fm, mat_G.t().repeat(curr_batch_size, 1, 1))   # batch size x i_m x R
                    first_mat = first_mat + torch.sum(temp_fm, dim=0)
            else:
                for i in tqdm(range(0, self.tensor.num_nnz, args.tucker_batch_lossnz)):
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_nnz - i) 
                    assert(curr_batch_size > 1)
                        
                    first_idx = self.tensor.indices[0][i:i+curr_batch_size]   # bs'
                    first_idx = torch.tensor(first_idx, dtype=torch.long, device=self.device)  # bs'
                    last_idx = self.tensor.indices[-1][i:i+curr_batch_size]   # bs'
                    last_idx = torch.tensor(last_idx, dtype=torch.long, device=self.device)  # bs'
                    min_k = torch.min(last_idx)
                    max_k = torch.max(last_idx) + 1
                    curr_mapping = self.mapping[min_k:max_k, :]   # bs' x i_max
                    curr_U = self.centroids.data[curr_mapping] * self.U_mask[min_k:max_k]   # bs' x i_max x R
                    
                    XUSV = curr_U[last_idx - min_k, first_idx, :]  # bs' x R
                    XUSV = face_split(XUSV, self.S[last_idx, :])   # bs' x R^2

                    for m in range(1, self.tensor.order - 1):
                        if m == mode: continue
                        else:
                            curr_idx = self.tensor.indices[m][i:i+curr_batch_size]        
                            curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)  # bs'                            
                            XUSV = face_split(XUSV, self.V[m-1][curr_idx, :])                                                                 # XUSV: bs' x R^(d-1)

                    curr_values = self.tensor.values[i:i+curr_batch_size]   # bs'
                    curr_values = torch.tensor(curr_values, dtype=torch.double, device=self.device)  # bs'
                    XUSV = curr_values.unsqueeze(-1) * XUSV  # bs' x R^(d-1)
                    temp_first_mat = XUSV @ mat_G.t()   # bs' x R

                    curr_idx = self.tensor.indices[mode][i:i+curr_batch_size]  # bs'
                    curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)  # bs'
                    first_mat = first_mat.index_add_(0, curr_idx, temp_first_mat)  # j_m x R
                    
            # second mat
            second_mat = 0
            for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_alsnx)):  
                curr_batch_size = min(self.tensor.num_tensor - i, args.tucker_batch_alsnx)
                assert(curr_batch_size > 1)
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
                curr_S = self.S[i:i+curr_batch_size, :]  # bs x R
                UtU = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U)  # bs x R x R
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
            for m in range(0, self.tensor.order-2):
                if m == 0:
                    Vkron = self.V[m].unsqueeze(0)
                else:
                    Vkron = batch_kron(Vkron, self.V[m].unsqueeze(0))
            Vkron = Vkron.squeeze()  # i_2*...i_m-1 x R^(d-2)                
            
            
            first_mat = torch.zeros((self.tensor.num_tensor, self.rank), dtype=torch.double, device=self.device)
            if args.is_dense:
                for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_lossnz)):                                  
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_tensor - i)
                    assert(curr_batch_size > 1)  
                    curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                    curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
                    # first mat

                    curr_tensor = self.tensor.src_tensor_torch[i:i+curr_batch_size].to(self.device)  # bs x i_max x i_2 x ... x i_(m-1)                                          
                    curr_tensor = torch.reshape(curr_tensor, (curr_batch_size, -1))   # bs x i_1*i_2* ... *i_(d-1) 
                    temp_fm = batch_kron(curr_U, Vkron.repeat(curr_batch_size, 1, 1))  # batch size x i_1*i_2*...i_m-1 x R^(d-1)
                    temp_fm = torch.bmm(curr_tensor.unsqueeze(1), temp_fm)   # batch size x 1 x R^(d-1)
                    first_mat[i:i+curr_batch_size, :] = temp_fm.squeeze() @ mat_G.t()  # batch size x R         
                                        
            else:
                for i in tqdm(range(0, self.tensor.num_nnz, args.tucker_batch_lossnz)):    
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_nnz - i)
                    assert(curr_batch_size > 1)
                    
                    first_idx = self.tensor.indices[0][i:i+curr_batch_size]   # bs'
                    first_idx = torch.tensor(first_idx, dtype=torch.long, device=self.device)  # bs'
                    last_idx = self.tensor.indices[-1][i:i+curr_batch_size]   # bs'
                    last_idx = torch.tensor(last_idx, dtype=torch.long, device=self.device)  # bs'
                    
                    min_k = torch.min(last_idx)
                    max_k = torch.max(last_idx) + 1
                    curr_mapping = self.mapping[min_k:max_k, :]   # k dist x i_max
                    curr_U = self.centroids.data[curr_mapping] * self.U_mask[min_k:max_k]   # bs' x i_max x R
                    XUVG = curr_U[last_idx-min_k, first_idx, :]   # bs' x rank
                    for m in range(1, self.tensor.order-1):
                        curr_idx = self.tensor.indices[m][i:i+curr_batch_size]
                        curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)
                        curr_V = self.V[m-1].data[curr_idx, :]   # bs' x rank                        
                        XUVG = face_split(XUVG, curr_V)
                    # bs' x R^(d-1)
                    
                    curr_values = self.tensor.values[i:i+curr_batch_size]   # bs'
                    curr_values = torch.tensor(curr_values, dtype=torch.double, device=self.device)  # bs'
                    XUVG = curr_values.unsqueeze(-1) * XUVG   # bs' x R^(d-1)
                    XUVG = XUVG @ mat_G.t()   # bs' x R              
                    first_mat = first_mat.index_add_(0, last_idx, XUVG)  # bs x rank
            
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
            for i in tqdm(range(self.tensor.num_tensor)):                       
                curr_mapping = self.mapping[i, :][:self.tensor.first_dim[i]]   # num_clusters
                second_mat[i] = torch.sum(GUVG[curr_mapping], dim=0)
            
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
                for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_lossnz)):                
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_tensor - i)               
                    assert(curr_batch_size > 1)
                    curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                    curr_U = self.centroids[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R    
                    
                    # Build tensor
                    curr_tensor = self.tensor.src_tensor_torch[i:i+curr_batch_size].to(self.device)  # bs x i_max x i_2 x ... x i_(m-1)                      
                    temp_perm = tuple([0] + [m for m in range(2, self.tensor.order)] + [1])
                    curr_tensor = torch.permute(curr_tensor, temp_perm)
                    curr_tensor = torch.reshape(curr_tensor, (curr_batch_size, -1, self.tensor.max_first))  # bs x i_2...i_(m-1) x i_1
                    curr_S = self.S.data[i:i+curr_batch_size, :]   # bs x rank
                    XUS = batch_kron(curr_U, curr_S.unsqueeze(1))    # bs x i_1 x rank^2                    
                    XUS = torch.bmm(curr_tensor, XUS) # bs x i_2...i_(m-1) x rank^2               
                    second_mat = second_mat + torch.sum(XUS, dim=0)   # i_2...i_(m-1) x rank^d                    
                second_mat = Vkron.t() @ second_mat  #  R^(d-2) x R^2
            else:    
                for i in tqdm(range(0, self.tensor.num_nnz, args.tucker_batch_lossnz)):                
                    curr_batch_size = min(args.tucker_batch_lossnz, self.tensor.num_nnz - i)                                                       
                    curr_fi = self.tensor.indices[0][i:i+curr_batch_size]
                    curr_li = self.tensor.indices[-1][i:i+curr_batch_size]
                    curr_fi = torch.tensor(curr_fi, dtype=torch.long, device=self.device)
                    curr_li = torch.tensor(curr_li, dtype=torch.long, device=self.device)
                    min_k = torch.min(curr_li)
                    max_k = torch.max(curr_li) + 1
                    
                    curr_mapping = self.mapping[min_k:max_k, :]   # k dist x i_max
                    curr_U = self.centroids[curr_mapping] * self.U_mask[min_k:max_k, :, :]   # k dist x i_max x R                
                    
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

                    US = curr_U.data[curr_li - min_k, curr_fi, :]   # bs' x rank 
                    US = face_split(US, self.S[curr_li, :].data)  # bs' x rank^2                                                            
                    temp_sm = VX.unsqueeze(-1) * US.unsqueeze(1)
                    second_mat = second_mat + torch.sum(temp_sm, dim=0)  # R^(d-2) x R^2
                                    
            # Third mat
            US = 0
            for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_alsnx)):
                curr_batch_size = min(args.tucker_batch_alsnx, self.tensor.num_tensor - i)                
                assert(curr_batch_size > 1)
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids[curr_mapping].data * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R       
                curr_S = self.S.data[i:i+curr_batch_size, :]    # batch size x rank
                UtU = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U)    # batch size x rank x rank
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
            sq_loss = self.L2_loss_tucker_dense(args.tucker_batch_lossnz)
        else:
            sq_loss = self.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
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
                sq_loss = self.L2_loss_tucker_dense(args.tucker_batch_lossnz)
            else:
                sq_loss = self.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
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
        