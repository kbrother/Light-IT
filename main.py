import argparse
from parafac2 import parafac2
from data import irregular_tensor
import torch
import gc
import os
import random
import numpy as np

# python main.py test_loss -tp ../data/23-Irregular-Tensor/test.pickle -de 1 -r 10 -d False
# python main.py test_loss -tp ../data/23-Irregular-Tensor/test.npy -de 1 -r 10 -d True
# python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious -r 5 -d False -de 0 -e 10 -lr 0.1
# python main.py train -tp ../data/23-Irregular-Tensor/cms.pickle -op results/cms -r 5 -d False -de 0 -e 10 -lr 0.1
# python main.py train -tp ../data/23-Irregular-Tensor/action.npy -op results/action -r 5 -d True -de 0 -e 10 -lr 0.1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='type of running')
    parser.add_argument('-tp', "--tensor_path", type=str)
    parser.add_argument('-op', "--output_path", type=str)
    parser.add_argument('-r', "--rank", type=int)
    parser.add_argument('-d', "--is_dense", type=str, default="error")
        
    parser.add_argument(
        "-de", "--device",
        action="store", type=int
    )    
    
    parser.add_argument(
        "-b", "--batch_size",
        action="store", default=2**22, type=int
    )
    
    parser.add_argument(
        "-e", "--epoch",
        action="store", type=int
    )
    
    parser.add_argument(
        "-eh", "--epoch_als",
        action="store", type=int, default=500
    )
    
    parser.add_argument(
        "-lr", "--lr", action="store", type=float
    )
        
    parser.add_argument(
        "-cb", "--cluster_batch",
        action="store", default=64, type=int
    )
    
    parser.add_argument(
        "-tbz", "--tucker_batch_lossz",
        action="store", default=2**6, type=int
    )
    
    parser.add_argument(
        "-tbn", "--tucker_batch_lossnz",
        action="store", default=2**9, type=int
    )
    
    parser.add_argument(
        "-tbu", "--tucker_batch_u",
        action="store", default=2**5, type=int
    )
        
    parser.add_argument(
        "-tbv", "--tucker_batch_v",
        action="store", default=2**6, type=int
    )
    
    parser.add_argument(
        "-tbs", "--tucker_batch_s",
        action="store", default=2**2, type=int
    )
    
    parser.add_argument(
        "-tbg", "--tucker_batch_g",
        action="store", default=2**6, type=int
    )
    
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    
    args = parser.parse_args()    
    if args.is_dense == "True":
        args.is_dense = True
    elif args.is_dense == "False":
        args.is_dense = False
    else:
        assert("wrong input")
    
    device = torch.device("cuda:" + str(args.device))
    _tensor = irregular_tensor(args.tensor_path, args.is_dense)
    print("load finish")
    
    _parafac2 = parafac2(_tensor, device, args)        
    if args.action == "train":
        _parafac2.quantization(args)        
        _parafac2.als(args)
    elif args.action == "test_loss":
        with torch.no_grad():
            if args.is_dense:
                print(f'dense: {_parafac2.L2_loss_dense(False, args.batch_size, _parafac2.U)}')
            else:
                print(f'sparse: {_parafac2.L2_loss(False, args.batch_size, _parafac2.U)}')            
    elif args.action == "test_tucker_loss":
        _parafac2.init_tucker(args)        
        _parafac2.G = torch.rand([_parafac2.rank]*_tensor.mode, device=_parafac2.device, dtype=torch.double)   
        if args.is_dense:
            sq_loss = _parafac2.L2_loss_tucker_dense(args.tucker_batch_lossnz)
            print(f'dense: {sq_loss}')
        else:
            sq_loss = _parafac2.L2_loss_tucker(args.tucker_batch_lossz, args.tucker_batch_lossnz)
            print(f'sparse: {sq_loss}')