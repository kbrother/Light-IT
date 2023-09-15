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
    #print(input1.shape)
    #print(input2.shape)
    return input1 * input2


class parafac2:
    
    def init_factor(self, batch_size):
        with torch.no_grad():
            _H = torch.rand((self.rank, self.rank), device=self.device, dtype=torch.double)
            for i in tqdm(range(0, self.tensor.k, batch_size)):
                curr_batch_size = min(batch_size, self.tensor.k - i)        
                # build tensor
                curr_rows = list(itertools.chain.from_iterable(self.tensor.rows_list[i: i+curr_batch_size]))
                curr_cols = list(itertools.chain.from_iterable(self.tensor.cols_list[i: i+curr_batch_size]))
                curr_heights = list(itertools.chain.from_iterable(self.tensor.heights_list[i: i+curr_batch_size]))
                curr_heights = [_h - i for _h in curr_heights]
                curr_vals = list(itertools.chain.from_iterable(self.tensor.vals_list[i: i+curr_batch_size]))                
                curr_tensor = torch.sparse_coo_tensor([curr_heights, curr_rows, curr_cols], curr_vals, (curr_batch_size, self.tensor.i_max, self.tensor.j), device=self.device, dtype=torch.double)  # batch size x i_max x j
                
                # compute SVD
                XV = torch.bmm(curr_tensor, self.V.repeat(curr_batch_size, 1, 1))  # batch size x i_max x R
                XVS = XV * self.S[i:i+curr_batch_size, :].unsqueeze(1)   # batch size x i_max x R
                XVSH = torch.bmm(XVS, _H.repeat(curr_batch_size, 1, 1))   # batc hsize x i_max x R
                Z, Sigma, Ph = torch.linalg.svd(XVSH, full_matrices=False)  # Z: batch size x i_max x R, Ph: bathc size x R x R
                self.U[i:i+curr_batch_size, :] = torch.bmm(Z, Ph) # batch size x i_max x rank

            # Normalize entry 
            _lambda = torch.sum(torch.square(_H), dim=0)  # R
            _H = _H / _lambda.unsqueeze(0)
            _lambda = _lambda * torch.sum(torch.square(self.V), dim=0)
            self.V = self.V / torch.sum(torch.square(self.V), dim=0).unsqueeze(0)
            _lambda = _lambda * torch.sum(torch.square(self.V), dim=0)
            self.S = self.S / torch.sum(torch.square(self.S), dim=0).unsqueeze(0)
            
            _lambda = torch.pow(_lambda, 1/3)
            _H = _H * _lambda.unsqueeze(0)
            self.V = self.V * _lambda.unsqueeze(0)
            self.S = self.S * _lambda.unsqueeze(0)
            
            # Normalize U
            self.U = torch.bmm(self.U, _H.repeat(self.tensor.k, 1, 1))
            

    def __init__(self, _tensor, device, args):      
        # Intialization
        self.device = device
        self.tensor = _tensor
        self.U_mask = torch.zeros((_tensor.k, _tensor.i_max, args.rank), device=device, dtype=torch.double)   # k x i_max x rank, 1 means the valid area
        for _k in range(_tensor.k):
            self.U_mask[_k, :_tensor.i[_k], :] = 1

        if args.factor_path is not None:
            factor_mat = loadmat(args.factor_path)
            _U = factor_mat['U'][0, :]
            self.S, self.V = factor_mat['S'], factor_mat['V']
            self.rank = self.S.shape[1]        

            self.U = np.zeros((_tensor.k, _tensor.i_max, self.rank))   # k x i_max x rank
            for _k in range(_tensor.k):
                self.U[_k, :_tensor.i[_k], :] = _U[_k]                
            self.U = torch.tensor(self.U, device=device)
            self.S, self.V = torch.tensor(self.S, device=device), torch.tensor(self.V, device=device)  
        else:
            self.rank = args.rank
            self.U = torch.rand((_tensor.k, _tensor.i_max, self.rank), device=device, dtype=torch.double)  # k x i_max x rank        
            self.U = self.U * self.U_mask            
            self.S = torch.rand((_tensor.k, self.rank), device=device, dtype=torch.double)      # k x rank
            self.V = torch.rand((_tensor.j, self.rank), device=device, dtype=torch.double)   # j x rank            
            self.init_factor(args.batch_init_factor)
            
        # Upload to gpu        
        self.centroids = torch.rand((_tensor.i_max, self.rank), device=device, dtype=torch.double)    # cluster centers,  i_max x rank
        self.centroids = torch.nn.Parameter(self.centroids)                      
        self.U = torch.nn.Parameter(self.U)
        self.S, self.V = torch.nn.Parameter(self.S), torch.nn.Parameter(self.V)                
        with torch.no_grad():
            sq_loss = self.L2_loss(False, args.batch_size, self.U * self.U_mask)
            print(f'fitness: {1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)}') 
            print(f'square loss: {sq_loss}')
        

    '''
        input_U: k x i_max x rank
    '''
    def L2_loss(self, is_train, batch_size, input_U):        
        # V * sigma * H^T
        # Correct non-zero terms
        _loss = 0        
        for i in range(0, self.tensor.k, batch_size//self.tensor.k):
            curr_batch_size = min(batch_size//self.tensor.k, self.tensor.k - i)
            curr_V = self.V.repeat(curr_batch_size, 1, 1)  # k x j x rank
            VS = curr_V * self.S[i:i+curr_batch_size, :].unsqueeze(1)   # k x j x rank        
            VS_t = torch.transpose(VS, 1, 2)  # k x rank x j  
            temp_tensor = torch.bmm(VS_t, VS)  # k x rank x rank
        
            # Computing the sum of square of tensors generated by parafac2
            curr_U = input_U[i:i+curr_batch_size, :, :]   # k x i_max x rank
            curr_mask = self.U_mask[i:i+curr_batch_size, :, :]
            temp_tensor = torch.matmul(curr_U*curr_mask, temp_tensor)    # k x i_max x rank
            sq_sum = torch.sum(temp_tensor * curr_U)   # k x i_max x rank
            if is_train: sq_sum.backward()
            _loss += sq_sum.item()              
        
        
        # Correct non-zero terms        
        for i in range(0, self.tensor.num_nnz, batch_size):
            sq_err = 0
            if self.tensor.num_nnz - i < batch_size:
                curr_batch_size = self.tensor.num_nnz - i
            else:
                curr_batch_size = batch_size
                
            # Prepare matrices
            _row = self.tensor.rows[i:i+curr_batch_size]
            _col = self.tensor.cols[i:i+curr_batch_size]
            _height = self.tensor.heights[i:i+curr_batch_size]
            _vals = self.tensor.vals[i:i+curr_batch_size]
            
            _U = input_U[_height, _row, :]  # batch size x rank            
            _S = self.S[_height, :]        # batch size x rank
            _V = self.V[_col, :]           # batch size x rank
            _approx = torch.sum(_U*_S*_V, dim=1)
            sq_err = sq_err - torch.sum(torch.square(_approx))
            sq_err = sq_err + torch.sum(torch.square(_approx - _vals))
            if is_train: sq_err.backward()
            _loss += sq_err.item()
            
        return _loss

    '''
        cluster_label: k x i_max
    '''
    def clustering(self):
        # Clustering
        with torch.no_grad():
            dist = torch.zeros((self.tensor.i_max, self.tensor.k, self.tensor.i_max), device=self.device)    
            for i in range(self.tensor.i_max):
                curr_dist = self.U - self.centroids[i,:].unsqueeze(0).unsqueeze(0) # k x i_max x rank
                curr_dist = torch.sum(torch.square(curr_dist), dim=-1) # k x i_max
                dist[i,:,:] = curr_dist
            #dist = self.Q.repeat(self.tensor.i_max, 1, 1, 1) - self.centroids.unsqueeze(1).unsqueeze(1)  # i_max x k x i_max x rank
            #dist = torch.sum(torch.square(dist), dim=-1) # i_max x k x i_max
            cluster_label = torch.argmin(dist, dim=0)  # k x i_max
        return cluster_label
        
        
    def quantization(self, args):
        optimizer = torch.optim.Adam([self.U, self.S, self.V, self.centroids], lr=args.lr)
        max_fitness = -100
        for _epoch in tqdm(range(args.epoch)):
            optimizer.zero_grad()
            # Clustering     
            U_clustered = self.centroids[self.clustering()]    # k x i_max x rank
            U_clustered = U_clustered * self.U_mask    
            sg_part = (self.U - U_clustered).detach()
            U_tricked = self.U - sg_part # refer to paper   # k x i_max x rank            
            self.L2_loss(True, args.batch_size, U_tricked) 
            cluster_loss = torch.sum(torch.square(U_clustered - (self.U * self.U_mask).detach()))            
            cluster_loss.backward()
            optimizer.step()
            
            if (_epoch + 1) % 10 == 0:
                with torch.no_grad():
                    U_clustered = self.centroids[self.clustering()]    # k x i_max x rank
                    U_clustered = U_clustered * self.U_mask
                    _loss = self.L2_loss(False, args.batch_size, U_clustered)              
                    _fitness = 1 - math.sqrt(_loss)/math.sqrt(self.tensor.sq_sum)
                    print(f'epoch: {_epoch}, l2 loss: {_loss}, fitness: {_fitness}')
                    with open(args.output_path + ".txt", 'a') as f:
                        f.write(f'epoch: {_epoch}, l2 loss: {_loss}, fitness: {_fitness}\n')
                       
                    if _fitness > max_fitness:
                        max_fitness = _fitness
                        final_U = self.U.data.clone().detach().cpu()
                        final_cents = self.centroids.data.clone().detach().cpu()
                        final_V = self.V.data.clone().detach().cpu()
                        final_S = self.S.data.clone().detach().cpu()
                        
        self.centroids.data.copy_(final_cents.to(self.device))
        self.U.data.copy_(final_U.to(self.device))
        self.V.data.copy_(final_V.to(self.device))
        self.S.data.copy_(final_S.to(self.device))
                   
            
    '''
        Use svd to initialize tucker decomposition
    '''
    def init_tucker(self, args):
        self.mapping = self.clustering()  # k x i_max
        self.mapping_mask = torch.zeros(self.tensor.k, self.tensor.i_max, dtype=torch.bool, device=self.device)   # k x i_max
        for _k in range(self.tensor.k):        
            self.mapping_mask[_k, :self.tensor.i[_k]] = True
        self.G = torch.zeros(self.rank, self.rank, self.rank, device=self.device, dtype=torch.double)
        idx_list = list(range(self.rank))
        self.G[idx_list, idx_list, idx_list] = 1                     
        
        
    '''
        input_U: k x i_max x rank
        _row, _col, _height: batch size
        return value: batch size
    '''
    def compute_irregular_tucker(self, input_U, _row, _col, _height):
        _U = input_U[_height, _row, :]  # batch size x rank
        _S = self.S[_height, :]        # batch size x rank
        _V = self.V[_col, :]           # batch size x rank
        _approx = torch.bmm(_U.unsqueeze(-1) , _V.unsqueeze(1))   # batch size x rank x rank
        #print(_approx.unsqueeze(-1).shape)
        batch_size = _row.shape[0]
        #print(_S.unsqueeze(1).unsqueeze(1).expand(batch_size, self.rank, 1, self.rank).shape)
        _approx = torch.matmul(_approx.unsqueeze(-1), _S.unsqueeze(1).unsqueeze(1).expand(batch_size, self.rank, 1, self.rank))    #  batch size x rank x rank x rank
        _approx = torch.sum(_approx * self.G.unsqueeze(0), (1, 2, 3))    # batch size        
        return _approx
    
    
    def L2_loss_tucker(self, batch_loss_zero, batch_loss_nz):        
        with torch.no_grad():
            input_U = self.centroids[self.mapping] * self.U_mask

            # compute zero terms        
            VtV = torch.mm(self.V.t(), self.V)   # rank x rank 
            StS = torch.matmul(self.S.unsqueeze(-1), self.S.unsqueeze(1))  # k x rank x rank
            mat_G = self.G.reshape(self.rank, self.rank**2)   # rank x rank^2

            num_kron_entry = self.tensor.k * self.rank**6              
            sq_loss = 0       
            for i in tqdm(range(0, self.tensor.k, batch_loss_zero)):
                curr_num_k = min(self.tensor.k - i, batch_loss_zero)
                UG = torch.bmm(input_U[i:i+curr_num_k, :, :], mat_G.repeat(curr_num_k, 1, 1))   # batch size x i_max x rank^2          
                middle_mat = batch_kron(VtV.repeat(curr_num_k, 1, 1), StS[i:i+curr_num_k, :, :])   # batch size x rank^2 x rank^2
                middle_mat = torch.bmm(UG, middle_mat)   # batch size x i_max x rank^2
                sq_loss = sq_loss + torch.sum(middle_mat * UG)  # batch size

            del UG, middle_mat, VtV, StS                      
            # Correct non-zero terms

            for i in tqdm(range(0, self.tensor.num_nnz, batch_loss_nz)):
                curr_batch_size = min(self.tensor.num_nnz - i, batch_loss_nz)                
                # Prepare matrices
                _row = self.tensor.rows[i:i+curr_batch_size].to(self.device)
                _col = self.tensor.cols[i:i+curr_batch_size].to(self.device)
                _height = self.tensor.heights[i:i+curr_batch_size].to(self.device)
                _vals = self.tensor.vals[i:i+curr_batch_size].to(self.device)

                _approx = self.compute_irregular_tucker(input_U, _row, _col, _height)
                curr_loss = torch.sum(torch.square(_approx - _vals)) - torch.sum(torch.square(_approx))                
                sq_loss = sq_loss + curr_loss
                
        return sq_loss        
        
    
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
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
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
            VtV = torch.mm(self.V.data.t(), self.V.data)   # R x R         
            first_mat = torch.linalg.pinv(VtV)   # Rx R
            second_mat, third_mat = 0, 0           
            for i in tqdm(range(0, self.tensor.k, args.tucker_batch_g)):
                curr_batch_size = min(args.tucker_batch_g, self.tensor.k - i)
                # Build tensor
                curr_rows = list(itertools.chain.from_iterable(self.tensor.rows_list[i: i+curr_batch_size]))
                curr_cols = list(itertools.chain.from_iterable(self.tensor.cols_list[i: i+curr_batch_size]))
                curr_heights = list(itertools.chain.from_iterable(self.tensor.heights_list[i: i+curr_batch_size]))
                curr_heights = [_h - i for _h in curr_heights]
                curr_vals = list(itertools.chain.from_iterable(self.tensor.vals_list[i: i+curr_batch_size]))       
                      
                curr_tensor = torch.sparse_coo_tensor([curr_heights, curr_cols, curr_rows], curr_vals, (curr_batch_size, self.tensor.j, self.tensor.i_max), device=self.device, dtype=torch.double)  # batch size x j x i_max
                
                # second mat
                curr_mapping = self.mapping[i:i+curr_batch_size, :]   # batch size x i_max
                curr_U = self.centroids.data[curr_mapping] * self.U_mask[i:i+curr_batch_size, :, :]   # batch size x i_max x R
                curr_S = self.S.data[i:i+curr_batch_size, :].unsqueeze(1)   # batch size x 1 x R
                US = batch_kron(curr_U, curr_S)   # batch size x i_max x R^2
                XUS = torch.bmm(curr_tensor, US)   # batch size x j x R^2
                second_mat = second_mat + torch.sum(XUS, dim=0)   # j x R^2
                
                # third mat
                UtU = torch.bmm(torch.transpose(curr_U, 1, 2), curr_U)  # batch size x R x R
                StS = torch.bmm(torch.transpose(curr_S, 1, 2), curr_S)  # batch size x R x R
                third_mat = third_mat + torch.sum(batch_kron(UtU, StS), dim=0)  # R^2 x R^2
            
            second_mat = torch.mm(self.V.data.t(), second_mat)  # R x R^2
            third_mat = torch.linalg.pinv(third_mat)   # R^2 x R^2
            self.G.data = torch.mm(first_mat, torch.mm(second_mat, third_mat))               
            self.G.data = torch.reshape(self.G.data, (self.rank, self.rank, self.rank))
            self.G.data = torch.permute(self.G.data, (1, 0, 2))
            
    
    def als(self, args):
        self.init_tucker(args)
        sq_loss = self.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
        prev_fit = 1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)
        print(f'loss after loaded:{prev_fit}')
        
        for e in range(args.epoch_als):
            self.als_G(args)        
            self.als_U(args)
            self.als_G(args)            
            self.als_V(args)
            self.als_G(args)       
            #sq_loss = self.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
            #curr_fit = 1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)   
            #print(f'epoch: {e+1}, before s:{curr_fit}')
            self.als_S(args)
                        
            sq_loss = self.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
            curr_fit = 1 - math.sqrt(sq_loss)/math.sqrt(self.tensor.sq_sum)            
            with open(args.output_path + ".txt", 'a') as f:
                f.write(f'als epoch: {e+1}, after s:{curr_fit}\n')
            print(f'als epoch: {e+1}, after s:{curr_fit}\n')
            
            if curr_fit - prev_fit < 1e-4:                 
                break
            prev_fit = curr_fit
        
        torch.save({
            'fitness': curr_fit, 'mapping': self.mapping,
            'centroids': self.centroids, 'U': self.U.data,
            'S': self.S, 'V': self.V, 'G': self.G
        }, args.output_path + ".pt")