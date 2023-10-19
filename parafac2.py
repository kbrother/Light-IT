from scipy.io import loadmat
from data import irregular_tensor
import math
import torch
from tqdm import tqdm
import numpy as np
import gc
import itertools

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
                for m in range(1, self.tensor.mode-2):
                    Vprod = khatri_rao(Vprod, self.V[m])   # row2 * rows3 *.... x rank
                
            for i in tqdm(range(self.tensor.num_tensor)):                                
                # build tensor
                if is_dense:  
                    curr_first_dim = max(self.rank, self.tensor.first_dim[i]) 
                    curr_dims = [curr_first_dim] + list(self.tensor.middle_dim)
                    curr_tensor = torch.zeros(curr_dims, device=self.device, dtype=torch.double)
                    curr_tensor[:self.tensor.first_dim[i]] = torch.from_numpy(self.tensor.src_tensor[i]).to(self.device)
                    curr_tensor = torch.reshape(curr_tensor, (curr_first_dim, -1))  # i_max x ...       
                
                    VS = Vprod * self.S[i].unsqueeze(0) # row2 * rows3 *.... x rank
                    XVS = curr_tensor @ VS  # i_max x R                     
                else:   
                    curr_tidx = self.tensor.tidx2start[i]
                    next_tidx = self.tensor.tidx2start[i + 1]
                    _V = torch.ones((next_tidx - curr_tidx, self.rank), device=self.device, dtype=torch.double)   # curr nnz x rank
                    for m in range(self.tensor.mode - 2):
                        curr_idx = torch.tensor(self.tensor.indices[m + 1][curr_tidx:next_tidx], dtype=torch.long, device=self.device)
                        _V = _V * self.V[m][curr_idx, :]
                    
                    curr_idx = torch.tensor(self.tensor.indices[self.tensor.mode - 1][curr_tidx:next_tidx], dtype=torch.long, device=self.device)
                    VS = _V * self.S[curr_idx, :]   # curr nnz x rank
                    curr_X = torch.tensor(self.tensor.values[curr_tidx:next_tidx], device=self.device, dtype=torch.double)
                    XVS_raw = curr_X.unsqueeze(1) * VS  # curr nnz x rank                
                    XVS = torch.zeros((max(self.rank, self.tensor.first_dim[i]), self.rank), device=self.device, dtype=torch.double)  # i_curr x rank
                    
                    curr_idx = torch.tensor(self.tensor.indices[0][curr_tidx:next_tidx], dtype=torch.long, device=self.device)    # nnz
                    XVS = XVS.index_add_(0, curr_idx, XVS_raw)  # i_curr x rank                    
                
                # compute SVD                               
                XVSH = XVS @ _H.t()   # i_curr x R                                 
                Z, Sigma, Ph = torch.linalg.svd(XVSH, full_matrices=False)  # Z: i_max x R, Ph:  R x R
                self.U[i,:max(self.tensor.first_dim[i], self.rank),:] = torch.mm(Z, Ph) # i_max x rank

            # Normalize entry 
            _lambda = torch.sqrt(torch.sum(torch.square(_H), dim=0))  # R
            _H = _H / _lambda.unsqueeze(0)
            
            for m in range(self.tensor.mode-2):
                curr_lambda = torch.sqrt(torch.sum(torch.square(self.V[m]), dim=0))
                _lambda = _lambda * curr_lambda
                self.V[m] = self.V[m] / curr_lambda.unsqueeze(0)
                
            curr_lambda = torch.sqrt(torch.sum(torch.square(self.S), dim=0))
            _lambda = _lambda * curr_lambda
            self.S = self.S / curr_lambda.unsqueeze(0)
            
            _lambda = torch.pow(_lambda, 1/self.tensor.mode)
            _H = _H * _lambda.unsqueeze(0)
            for m in range(self.tensor.mode-2):
                self.V[m] = self.V[m] * _lambda.unsqueeze(0)
            self.S = self.S * _lambda.unsqueeze(0)
            
            # Normalize U
            self.U = torch.bmm(self.U, _H.repeat(self.tensor.num_tensor, 1, 1))
            
        UTU = torch.bmm(torch.transpose(self.U, 1, 2), self.U)   # k x rank x rank                
        HTH = _H.t() @ _H
        #iden = torch.diag(torch.ones(self.rank)).to(self.device)
        print(torch.mean(torch.abs(UTU - HTH.unsqueeze(0))))        
            

    def __init__(self, _tensor, device, args):      
        # Intialization
        self.device = device
        self.tensor = _tensor
        if args.is_dense:
            scale_factor = 1
        else:
            scale_factor = 0.01
        self.U_mask = torch.zeros((_tensor.num_tensor, _tensor.max_first, args.rank), device=device, dtype=torch.double)   # k x i_max x rank, 1 means the valid area
        for _k in range(_tensor.num_tensor):
            self.U_mask[_k, :_tensor.first_dim[_k], :] = 1

        self.rank = args.rank
        self.U = scale_factor * torch.rand((_tensor.num_tensor, _tensor.max_first, self.rank), device=device, dtype=torch.double)  # k x i_max x rank        
        self.U = self.U * self.U_mask          
        self.V = []
        for m in range(self.tensor.mode-2):
            curr_dim = _tensor.middle_dim[m]
            self.V.append(scale_factor * torch.rand((curr_dim, self.rank), device=device, dtype=torch.double))  # j x rank
        self.S = scale_factor * torch.rand((_tensor.num_tensor, self.rank), device=device, dtype=torch.double)      # k x rank        
        self.init_factor(args.is_dense)        

        # Upload to gpu        
        self.centroids = scale_factor * torch.rand((_tensor.max_first, self.rank), device=device, dtype=torch.double)    # cluster centers,  i_max x rank
        self.centroids = torch.nn.Parameter(self.centroids)                      
        self.U = torch.nn.Parameter(self.U)
        self.S = torch.nn.Parameter(self.S)
        for m in range(_tensor.mode-2):
            self.V[m] = torch.nn.Parameter(self.V[m])
        
        with torch.no_grad():
            if args.is_dense:
                sq_loss = self.L2_loss_dense(False, args.batch_size, self.U * self.U_mask)
            else:
                sq_loss = self.L2_loss(False, args.batch_size, self.U * self.U_mask)
            print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)}') 
            print(f'square loss: {sq_loss}')
            
            
    '''
        input_U: num_tensor x i_max x rank
    '''
    def L2_loss_dense(self, is_train, batch_size, input_U):
        _loss = 0
        Vprod = self.V[0]
        for i in range(1, self.tensor.mode-2):
            Vprod = khatri_rao(Vprod, self.V[i]).unsqueeze(0)   # 1 x i_2 * ... * i_(m-1) x rank        
            
        for i in range(0, self.tensor.num_tensor, batch_size):                        
            curr_batch_size = min(batch_size, self.tensor.num_tensor - i) 
            curr_S = self.S[i:i+curr_batch_size, :].unsqueeze(1)  # bs x 1 x rank
            VS = Vprod * curr_S   # bs x i_2 * ... * i_(m-1) x rank        
            VS = torch.transpose(VS, 1, 2)   # bs x rank x i_2 * ... * i_(m-1)        
            
            curr_U = input_U[i:i+curr_batch_size, :, :] # batch size x i_max x rank
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
        input_U: k x i_max x rank
    '''
    def L2_loss(self, is_train, batch_size, input_U):                
        # zero terms             
        VtV = torch.ones((self.rank, self.rank), device=self.device, dtype=torch.double)  # r x r
        for i in range(self.tensor.mode-2):
            VtV = VtV * (self.V[i].t() @ self.V[i])  # r x r
            
        UtU = torch.bmm(torch.transpose(input_U, 1, 2), input_U) # k x rank x rank
        StS = torch.bmm(self.S.unsqueeze(2), self.S.unsqueeze(1)) # k x rank x rank
        first_mat = torch.sum(UtU * StS, dim=0)  # rank x rank
        sq_sum = torch.sum(first_mat * VtV)
        if is_train: 
            sq_sum.backward()
        _loss = sq_sum.item()
        
        # Correct non-zero terms        
        for i in range(0, self.tensor.num_nnz, batch_size):
            curr_batch_size = min(batch_size, self.tensor.num_nnz - i)
            first_idx = torch.tensor(self.tensor.indices[0][i: i+curr_batch_size], device=self.device, dtype=torch.long) # bs
            final_idx = torch.tensor(self.tensor.indices[-1][i: i+curr_batch_size], device=self.device, dtype=torch.long)  # bs
            
            approx = input_U[final_idx, first_idx, :] * self.S[final_idx, :]  # bs x rank
            for m in range(1, self.tensor.mode-1):
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
        for _epoch in tqdm(range(args.epoch)):
            optimizer.zero_grad()
            # Clustering     
            U_clustered = self.centroids[self.clustering(args)]    # k x i_max x rank
            U_clustered = U_clustered * self.U_mask    
            sg_part = (self.U - U_clustered).detach()
            U_tricked = self.U - sg_part # refer to paper   # k x i_max x rank   
            if args.is_dense:
                self.L2_loss_dense(True, args.batch_size, U_tricked) 
            else:
                self.L2_loss(True, args.batch_size, U_tricked) 
            cluster_loss = torch.sum(torch.square(U_clustered - (self.U * self.U_mask).detach()))            
            cluster_loss.backward()
            optimizer.step()
            
            if (_epoch + 1) % 10 == 0:
                with torch.no_grad():
                    U_clustered = self.centroids[self.clustering(args)]    # k x i_max x rank
                    U_clustered = U_clustered * self.U_mask
                    if args.is_dense:
                        _loss = self.L2_loss_dense(False, args.batch_size, U_clustered) 
                    else:
                        _loss = self.L2_loss(False, args.batch_size, U_clustered)              
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
                        
        self.centroids.data.copy_(final_cents.to(self.device))
        self.U.data.copy_(final_U.to(self.device))
        
        for m in range(self.tensor.mode-2):
            self.V[m].data.copy_(final_V[m].to(self.device))
        self.S.data.copy_(final_S.to(self.device))
                
        torch.save({
            'fitness': max_fitness, 'centroids': self.centroids.data, 
            'U': self.U.data, 'S': self.S.data, 'V': final_V,
        }, args.output_path + "_cp.pt")
                   
            
    '''
        Use cpd to initialize tucker decomposition
    '''
    def init_tucker(self, args):
        self.mapping = self.clustering(args)  # k x i_max
        self.mapping_mask = torch.zeros(self.tensor.num_tensor, self.tensor.max_first, dtype=torch.bool, device=self.device)   # k x i_max
        for _k in range(self.tensor.num_tensor):        
            self.mapping_mask[_k, :self.tensor.first_dim[_k]] = True
        self.G = torch.zeros([self.rank]*self.tensor.mode, device=self.device, dtype=torch.double)    
        self.G = self.G.fill_diagonal_(1)
        
        
    '''
        input_U: k x i_max x rank        
        indices: list of indices        
        return value: batch size
    '''
    def compute_irregular_tucker(self, input_U, indices):
        first_mat = input_U[indices[-1], indices[0], :]  # batch size x rank
        for m in range(self.tensor.mode-2):            
            first_mat = face_split(first_mat, self.V[m][indices[m+1], :])  
        
        first_mat = face_split(first_mat, self.S[indices[-1], :])   # batch size x rank...
        vec_G = torch.flatten(self.G).unsqueeze(-1)  # rank..... x 1 
        approx = torch.mm(first_mat, vec_G).squeeze()            
        return approx
    
    
    def L2_loss_tucker(self, batch_loss_zero, batch_loss_nz):        
        with torch.no_grad():
            input_U = self.centroids[self.mapping] * self.U_mask  # k x i_max x rank

            # compute zero terms                    
            for m in range(self.tensor.mode-2):
                if m == 0: 
                    middle_mat = torch.mm(self.V[m].t(), self.V[m]).unsqueeze(0)
                else:
                    middle_mat = batch_kron(middle_mat, torch.mm(self.V[m].t(), self.V[m]).unsqueeze(0))                   
            middle_mat = middle_mat.squeeze()   # rank x rank
            StS = torch.matmul(self.S.unsqueeze(-1), self.S.unsqueeze(1))  # k x rank x rank
            mat_G = self.G.reshape(self.rank, -1)   # rank x rank^2             
            sq_loss = 0       
            
            for i in tqdm(range(0, self.tensor.num_tensor, batch_loss_zero)):
                curr_num_k = min(self.tensor.num_tensor - i, batch_loss_zero)
                UG = torch.bmm(input_U[i:i+curr_num_k, :, :], mat_G.repeat(curr_num_k, 1, 1))   # batch size x i_max x rank^2          
                curr_mmat = batch_kron(middle_mat.repeat(curr_num_k, 1, 1), StS[i:i+curr_num_k, :, :])   # batch size x rank^2 x rank^2
                #print(UG.shape)
                #print(curr_mmat.shape)
                curr_mmat = torch.bmm(UG, curr_mmat)   # batch size x i_max x rank^2
                sq_loss = sq_loss + torch.sum(curr_mmat * UG).item()  # batch size
            
            # Correct non-zero terms
            for i in tqdm(range(0, self.tensor.num_nnz, batch_loss_nz)):
                curr_batch_size = min(self.tensor.num_nnz - i, batch_loss_nz)                
                # Prepare matrices
                curr_indices = [torch.tensor(self.tensor.indices[m][i:i+curr_batch_size], dtype=torch.long, device=self.device) for m in range(self.tensor.mode)]                
                curr_vals = torch.tensor(self.tensor.values[i:i+curr_batch_size], dtype=torch.double, device=self.device)

                _approx = self.compute_irregular_tucker(input_U, curr_indices)
                curr_loss = torch.sum(torch.square(_approx - curr_vals)) - torch.sum(torch.square(_approx))              
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
        print(self.tensor.mode)
        for m in range(self.tensor.mode-2):
            if m == 0:
                VKron = self.V[m].data.unsqueeze(0)  # 1 x j_1 x R
            else:
                VKron = batch_kron(VKron, self.V[m].data.unsqueeze(0))              
        VKron = VKron.squeeze()    
        # j_1...j_(d-2) x R^(d-2)
            
        with torch.no_grad():        
            for i in tqdm(range(0, self.tensor.num_tensor, batch_size)):                
                curr_batch_size = min(batch_size, self.tensor.num_tensor - i)
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
            first_mat = torch.zeros(self.tensor.i_max, self.rank, device=self.device, dtype=torch.double)   # i_max x rank
            second_mat = torch.zeros(self.tensor.i_max, self.rank, self.rank, device=self.device, dtype=torch.double)    # i_max x rank x rank
            curr_G = torch.reshape(self.G, (self.rank, -1))   # rank x rank^2
            VtV = torch.mm(self.V.data.t(), self.V.data)   # R x R     

            for i in tqdm(range(0, self.tensor.k, args.tucker_batch_u)):
                curr_batch_size = min(args.tucker_batch_u, self.tensor.k - i)
                # declare tensor
                # Build tensor
                if args.is_dense:            
                    curr_tensor = self.set_curr_tensor(curr_batch_size, i)
                else:                    
                    curr_rows = list(itertools.chain.from_iterable(self.tensor.rows_list[i: i+curr_batch_size]))
                    curr_cols = list(itertools.chain.from_iterable(self.tensor.cols_list[i: i+curr_batch_size]))
                    curr_heights = list(itertools.chain.from_iterable(self.tensor.heights_list[i: i+curr_batch_size]))
                    curr_heights = [_h - i for _h in curr_heights]
                    curr_vals = list(itertools.chain.from_iterable(self.tensor.vals_list[i: i+curr_batch_size]))       

                    curr_tensor = torch.sparse_coo_tensor([ curr_heights, curr_rows, curr_cols], curr_vals, (curr_batch_size, self.tensor.i_max, self.tensor.j), device=self.device, dtype=torch.double)  # batch size x i_max x j
                
                # common features
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_mapping_mask = self.mapping_mask[i:i+curr_batch_size, :]   # batch size x i_max                

                # handle the first mat
                curr_S = self.S.data[i:i+curr_batch_size, :].unsqueeze(1)   # batch size x 1 x rank                
                VS = batch_kron(self.V.data.repeat(curr_batch_size, 1, 1), curr_S)   # batch size x J X rank^2
                VSG = torch.bmm(VS, curr_G.t().repeat(curr_batch_size, 1, 1))    # batch size x J x rank
                XVSG = torch.bmm(curr_tensor, VSG)   # batch size x i_max x rank
                temp_mapping = curr_mapping.unsqueeze(-1).expand(-1, -1, self.rank)   # batch size x i_max x rank
                temp_first_mat = torch.zeros(curr_batch_size, self.tensor.i_max, self.rank, device=self.device, dtype=torch.double)    # batch size x i_max x rank
                temp_first_mat = temp_first_mat.scatter_add_(1, temp_mapping, XVSG)     # batch size x i_max x rank
                first_mat = first_mat + torch.sum(temp_first_mat, dim=0)
                
                # handle the second mat
                SS = torch.bmm(torch.transpose(curr_S, 1, 2), curr_S)    # batch size x rank x rank 
                V2S2 = batch_kron(VtV.repeat(curr_batch_size, 1, 1), SS)   # batch size x rank^2 x rank^2
                GV2S2 = torch.bmm(curr_G.repeat(curr_batch_size, 1, 1), V2S2)  # batch size x rank x rank^2
                GV2S2G = torch.bmm(GV2S2, curr_G.t().repeat(curr_batch_size, 1, 1))  # batch size x rank x rank                
                GV2S2G = GV2S2G.unsqueeze(1).expand(-1, self.tensor.i_max, -1, -1)   # batch size x i_max x rank x rank          
                GV2S2G = torch.reshape(GV2S2G[curr_mapping_mask, :, :], (-1, self.rank, self.rank))   # small batch x rank x rank                
                
                curr_mapping = curr_mapping[curr_mapping_mask]  # small batch
                red_batch_size = curr_mapping.shape[0]
                temp_second_mat = torch.zeros(self.tensor.i_max, self.rank, self.rank, device=self.device, dtype=torch.double)   # i_max x rank x rank
                temp_second_mat = temp_second_mat.index_add_(0, curr_mapping, GV2S2G)   # i_max x rank x rank
                second_mat = second_mat + temp_second_mat
                
            self.centroids.data = torch.bmm(first_mat.unsqueeze(1), torch.linalg.pinv(second_mat)).squeeze()              
            
            
    def als_V(self, args):    
        #print(U_clustered.shape)        
        with torch.no_grad():
            mat_G = torch.reshape(torch.transpose(self.G.data, 0, 1), (self.rank, -1))   # R x R^2
            front_mat, end_mat = 0, 0
            for i in tqdm(range(0, self.tensor.k, args.tucker_batch_v)):
                curr_batch_size = min(args.tucker_batch_v, self.tensor.k - i)        
                if args.is_dense:
                    curr_tensor = self.set_curr_tensor(curr_batch_size, i)
                    curr_tensor = torch.transpose(curr_tensor, 1, 2)   # batch size x j x i_max
                else:                                                
                    # Build tensor
                    curr_rows = list(itertools.chain.from_iterable(self.tensor.rows_list[i: i+curr_batch_size]))
                    curr_cols = list(itertools.chain.from_iterable(self.tensor.cols_list[i: i+curr_batch_size]))
                    curr_heights = list(itertools.chain.from_iterable(self.tensor.heights_list[i: i+curr_batch_size]))
                    curr_heights = [_h - i for _h in curr_heights]
                    curr_vals = list(itertools.chain.from_iterable(self.tensor.vals_list[i: i+curr_batch_size]))                
                    curr_tensor = torch.sparse_coo_tensor([curr_heights, curr_cols, curr_rows], curr_vals, (curr_batch_size, self.tensor.j, self.tensor.i_max), device=self.device, dtype=torch.double)  # batch size x j x i_max

                # Build front mat
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R                
                curr_S = self.S.data[i:i+curr_batch_size, :].unsqueeze(1)  # batch size x 1 x R
                US = batch_kron(curr_U, curr_S)  # batch size x i_max x R^2
                curr_front_mat = torch.bmm(curr_tensor, US)    # batch size x j x R^2
                front_mat = front_mat + torch.sum(torch.bmm(curr_front_mat, torch.transpose(mat_G, 0, 1).repeat(curr_batch_size, 1, 1,)), dim=0)   # j x R

                # Build end mat                  # batch size x i_max x R
                UtU = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U)  # batch size x R x R
                StS = torch.bmm(torch.transpose(curr_S, 1, 2), curr_S) # batch size x R x R
                end_mat = end_mat + torch.sum(batch_kron(UtU, StS) , dim=0)   # R^2 x R^2
            
            end_mat = torch.mm(torch.mm(mat_G, end_mat), mat_G.t())   # R X R
            end_mat = torch.linalg.pinv(end_mat)
            self.V.data = torch.mm(front_mat, end_mat)
            
                
    def als_S(self, args):        
        with torch.no_grad():
            mat_G = torch.reshape(torch.permute(self.G.data, (2, 0, 1)), (self.rank, -1))   # R x R^2
            VtV = torch.mm(self.V.data.t(), self.V.data)   # R x R
            for i in tqdm(range(0, self.tensor.k, args.tucker_batch_s)):
                curr_batch_size = min(args.tucker_batch_s, self.tensor.k - i)
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
                    
                if args.is_dense:
                    curr_tensor = self.set_curr_tensor(curr_batch_size, i)  #  batch size x i_max x j   
                    curr_tensor = curr_tensor.view(curr_batch_size, -1).unsqueeze(1)    # batch size x 1 x i_max*j
                    curr_V = self.V.data.unsqueeze(0).repeat(curr_batch_size, 1, 1)  # batch size x j x R
                    UV = batch_kron(curr_U, curr_V)  # batch size x i_max*j x R^2
                    XUV = torch.bmm(curr_tensor, UV)   # batch size x 1 x R^2
                    front_mat = torch.bmm(XUV, mat_G.t().repeat(curr_batch_size, 1, 1))  # batch size x 1 x R
                else:
                    # Build tensor
                    curr_rows = list(itertools.chain.from_iterable(self.tensor.rows_list[i: i+curr_batch_size]))
                    curr_cols = list(itertools.chain.from_iterable(self.tensor.cols_list[i: i+curr_batch_size]))
                    curr_idx = [curr_rows[ii]*self.tensor.j + curr_cols[ii] for ii in range(len(curr_rows))]
                    curr_heights = list(itertools.chain.from_iterable(self.tensor.heights_list[i: i+curr_batch_size]))
                    curr_heights = [_h - i for _h in curr_heights]
                    curr_heights = torch.tensor(curr_heights, dtype=torch.long, device=self.device)   # entry
                    curr_vals = list(itertools.chain.from_iterable(self.tensor.vals_list[i: i+curr_batch_size]))      
                    curr_vals = torch.tensor(curr_vals, dtype=torch.double, device=self.device)    # num_entry             

                    # front mat
                    front_mat = torch.zeros(curr_batch_size, self.rank**2, device=self.device, dtype=torch.double)   # batch size x rank^2                          
                    nnz_U = curr_U[curr_heights, curr_rows, :]     # num_entry x R
                    nnz_V = self.V.data[curr_cols, :]   # num_entry x R
                    UV = batch_kron(nnz_U.unsqueeze(1), nnz_V.unsqueeze(1)).squeeze()   # num_entry x R^2
                    XUV = UV * curr_vals.unsqueeze(1)      # num_entry x R^2
                    front_mat = front_mat.index_add_(0, curr_heights, XUV)   # batch size x rank^2      
                    front_mat = torch.bmm(front_mat.unsqueeze(1), mat_G.t().repeat(curr_batch_size, 1, 1))   # batch size x 1 x rank
                    
                # end mat
                UtU = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U)   # batch size x R x R
                end_mat = batch_kron(UtU, VtV.repeat(curr_batch_size, 1, 1))    # batch size x R^2 X R^2
                end_mat = torch.bmm(mat_G.repeat(curr_batch_size, 1, 1), end_mat)   # batch size x R x R^2
                end_mat = torch.bmm(end_mat, mat_G.t().repeat(curr_batch_size, 1, 1))   # batch size x R x R
                end_mat = torch.linalg.pinv(end_mat)
                
                self.S.data[i:i+curr_batch_size, :] = torch.bmm(front_mat, end_mat).squeeze()   
                
                
    def als_G(self, args):
        with torch.no_grad():            
            temp_V = self.V[0].data  # j_1 x R
            first_mat = torch.linalg.pinv(temp_V.t() @ temp_V)   # R x R
            second_mat = 0   # j_1 x R^(d-1)
            third_mat = 0    # R^(d-1) x R^(d-1)
            
            # define VtV
            if self.tensor.mode > 3:
                for m in range(1, self.tensor.mode-2):            
                    if m == 1:
                        VtV = self.V[m].t()@self.V[m]
                    else:
                        VtV = batch_kron(VtV, self.V[m].t()@self.V[m])            
                VtV = VtV.squeeze()
                # rank^(d-3) x rank^(d-3)

                # define Vkron
                if args.is_dense:
                    for m in range(1, self.tensor.mode-2):
                        if m == 1:
                            Vkron = self.V[m].unsqueeze(0)
                        else:
                            Vkron = batch_kron(Vkron, self.V[m].unsuqeeze(0))
                    Vkron = Vkron.squeeze()    
                    # j_2*...*j_(m-2) x rank^(d-3)
                
            for i in tqdm(range(0, self.tensor.num_tensor, args.tucker_batch_g)):
                curr_batch_size = min(args.tucker_batch_g, self.tensor.num_tensor - i)               
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
                curr_S = self.S.data[i:i+curr_batch_size, :].unsqueeze(1)   # batch size x 1 x R
                                
                if args.is_dense:
                    # Build tensor
                    curr_tensor = self.tensor.src_tensor_torch[i:i+curr_batch_size].to(self.device)  # bs x i_max x i_2 x ... x i_(m-1)  
                    curr_tensor = torch.transpose(curr_tensor, 1, 2)   # bs x i_2 x i_max x ... x i_(m-1)  
                    curr_tensor = torch.reshape(curr_tensor, (curr_batch_size, self.tensor.middle_dim[0], -1))   # bs x i_2 x ....
                    if self.tensor.mode >= 4:
                        UVS = batch_kron(curr_U, Vkron.repeat(curr_batch_size, 1, 1))  # batch_size x i_max*j_2*...*j_(m-2) x rank^(d-2)
                    else:
                        UVS = curr_U # batch size x i_max x rank
                    UVS = batch_kron(UVS, curr_S)   # batch size x i_max*j_2*...*j_(m-2) x rank^(d-1)
                    
                    XUVS = torch.bmm(curr_tensor, UVS)  # batch size x i_2 x rank^(d-1)
                    second_mat = second_mat + torch.sum(XUVS, dim=0)   #  i_2 x rank^(d-1)
                else: 
                    curr_tidx = self.tensor.tidx2start[i]
                    next_tidx = self.tensor.tidx2start[i + curr_batch_size]
                    curr_fi = self.tensor.indices[0][curr_tidx:next_tidx]
                    curr_li = self.tensor.indices[-1][curr_tidx:next_tidx]
                    curr_fi = torch.tensor(curr_fi, dtype=torch.long, device=self.device)
                    curr_li = torch.tensor(curr_li, dtype=torch.long, device=self.device)
                    
                    XUVS = curr_U.data[curr_li - i, curr_fi, :]   # bs' x rank
                    for m in range(2, self.tensor.mode-1):
                        curr_idx = self.tensor.indices[m][curr_tidx:next_tidx]
                        curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)
                        _V = self.V[m-1][curr_idx, :].data     # bs' x rank
                        XUVS = face_split(XUVS, _V)
                    # XUVS: bs' x rank^(d-2)
                    
                    XUVS = face_split(XUVS, self.S[curr_li, :].data)    # bs' x rank^(d-1)
                    curr_values = torch.tensor(self.tensor.values[curr_tidx:next_tidx], dtype=torch.float, device=self.device)   # bs'
                    XUVS = XUVS * curr_values.unsqueeze(-1)  # bs' x rank^(d-1)
                    curr_idx = self.tensor.indices[1][curr_tidx:next_tidx]
                    curr_idx = torch.tensor(curr_idx, dtype=torch.long, device=self.device)                                       
                    curr_idx = (curr_li - i)*self.tensor.middle_dim[0] + curr_idx   # bs'
                    temp_second_mat = torch.zeros((curr_batch_size*self.tensor.middle_dim[0], self.rank**(self.tensor.mode-1)), device=self.device, dtype=torch.double)   # bs*j_1 x R^(d-1)
                    
                    temp_second_mat = temp_second_mat.index_add_(0, curr_idx, XUVS)   #  # bs*j_1 x R^(d-1)
                    temp_second_mat = torch.reshape(temp_second_mat, (curr_batch_size, self.tensor.middle_dim[0], -1))                    
                    second_mat = second_mat + torch.sum(temp_second_mat, dim=0)   # j_1 x R^(d-1)
                
                # third mat
                temp_third_mat = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U)  # batch size x R x R                
                if self.tensor.mode > 3:
                    temp_third_mat = batch_kron(temp_third_mat, VtV.repeat(curr_batch_size, 1, 1)) # batch size x R^(d-2) x R^(d-2)                
                StS = torch.bmm(torch.transpose(curr_S, 1, 2), curr_S)  # batch size x R x R                             
                temp_third_mat = batch_kron(temp_third_mat, StS)   # batch size x R^(d-1) x R^(d-1)                
                third_mat = third_mat + torch.sum(temp_third_mat, dim=0)  # R^(d-1) x R^(d-1)
            
            second_mat = torch.mm(self.V[0].data.t(), second_mat)  # R x R^(d-1)
            third_mat = torch.linalg.pinv(third_mat)   # R^(d-1) x R^(d-1)
            self.G.data = first_mat @ second_mat @ third_mat  # R x R^(d-1)
            orig_shape = tuple(self.rank for _ in range(self.tensor.mode))
            self.G.data = torch.reshape(self.G.data, orig_shape)
            self.G.data = torch.transpose(self.G.data, 0, 1)
            
    
    def als(self, args):                
        self.init_tucker(args)        
        if args.is_dense:
            sq_loss = self.L2_loss_tucker_dense(args.tucker_batch_lossnz)
        else:
            sq_loss = self.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
        prev_fit = 1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)
        print(f'loss after loaded:{prev_fit}')
                
        for e in range(args.epoch_als):
            self.als_G(args)        
            #self.als_U(args)
            #self.als_G(args)            
            #self.als_V(args)
            #self.als_G(args)                   
            #self.als_S(args)
                        
            if args.is_dense:
                sq_loss = self.L2_loss_tucker_dense(args.batch_size)
            else:
                sq_loss = self.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
            curr_fit = 1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)            
            with open(args.output_path + ".txt", 'a') as f:
                f.write(f'als epoch: {e+1}, after s:{curr_fit}\n')
            print(f'als epoch: {e+1}, after s:{curr_fit}')
            
            if curr_fit - prev_fit < 1e-4:                 
                break
            prev_fit = curr_fit
        
        '''
        torch.save({
            'fitness': curr_fit, 'mapping': self.mapping,
            'centroids': self.centroids, 'U': self.U.data,
            'S': self.S, 'V': self.V, 'G': self.G
        }, args.output_path + ".pt")
        '''
        