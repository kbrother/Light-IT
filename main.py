import argparse
from parafac2 import parafac2
from data import irregular_tensor
import torch
import gc
import os
import random
import numpy as np
import time
import math
from parafac2 import clear_memory

# python main.py -tp ../data/23-Irregular-Tensor/cms.pickle -op results/cms_r30_lr0.1_s0 -r 30 -d False -de 0 -e 500 -lr 0.1 -s 0
# python main.py -tp ../data/23-Irregular-Tensor/action.npy -op results/action_r30_lr0.1_s0 -r 30 -d True -de 0 -e 500 -lr 0.1 -s 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('-tp', "--tensor_path", type=str)
    parser.add_argument('-op', "--output_path", type=str)
    parser.add_argument('-r', "--rank", type=int)
    parser.add_argument('-d', "--is_dense", type=str, default="error")
        
    parser.add_argument(
        "-de", "--device",
        action="store", type=int, default=0
    )    
    
    parser.add_argument(
        "-bz", "--batch_lossz",
        action="store", default=2**10, type=int
    )
    
    parser.add_argument(
        "-bnz", "--batch_lossnz",
        action="store", default=2**22, type=int
    )
    
    parser.add_argument(
        "-e", "--epoch",
        action="store", type=int
    )
    
    parser.add_argument(
        "-lr", "--lr", action="store", type=float
    )
    
    parser.add_argument(
        "-s", "--seed", 
        action="store", type=int,
    )

    parser.add_argument(
        "-cb", "--cluster_batch",
        action="store", default=128, type=int
    )

    args = parser.parse_args()    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.is_dense == "True":
        args.is_dense = True
    elif args.is_dense == "False":
        args.is_dense = False
    else:
        assert("wrong input")
    
    device = torch.device("cuda:" + str(args.device))
    _tensor = irregular_tensor(args.tensor_path, args.is_dense)
    print("load finish")
            

    start_time = time.time()
    _parafac2 = parafac2(_tensor, device, args)
    _parafac2.quantization(args)     
    with open(args.output_path + ".txt", 'a') as f:
        f.write(f'cp time: {time.time() - start_time}\n')