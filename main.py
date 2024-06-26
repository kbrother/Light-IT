import argparse
from model import LightIT
from data import irregular_tensor
import torch
import gc
import os
import random
import numpy as np
import time
import math
from model import clear_memory

# python main.py test_init -tp ../input/23-Irregular-Tensor/action.npy -de 1 -r 2 -d True -s 0
# python main.py test_init -tp ../input/23-Irregular-Tensor/cms.pickle -de 1 -r 2 -d False -s 0

# python main.py test_loss -tp ../input/23-Irregular-Tensor/test.pickle -de 1 -r 3 -d False -s 0
# python main.py test_loss -tp ../input/23-Irregular-Tensor/test.npy -de 1 -r 3 -d True -s 0

# python main.py train -tp ../input/23-Irregular-Tensor/delicious_small.pickle -op results/delicious -r 5 -d False -de 4 -e 10 -lr 0.01 -ea 5
# python main.py train -tp ../input/23-Irregular-Tensor/cms_small.pickle -op results/cms -r 5 -d False -de 4 -e 10 -lr 0.1 -ea 5
# python main.py train -tp ../input/23-Irregular-Tensor/action.npy -op results/action -r 5 -d True -de 4 -e 10 -lr 0.1 -ea 5 -tbnz 50
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='type of running')
    parser.add_argument('-tp', "--tensor_path", type=str)
    parser.add_argument('-op', "--output_path", type=str)
    parser.add_argument('-r', "--rank", type=int)
    parser.add_argument('-d', "--is_dense", type=str, default="error")
        
    parser.add_argument(
        "-de", "--device",
        action="store", type=int, default=0
    )    
    
    parser.add_argument(
        "-b", "--batch",
        action="store", default=2**10, type=int
    )
    
    parser.add_argument(
        "-bnz", "--batch_nz",
        action="store", default=2**22, type=int
    )
    
    parser.add_argument(
        "-e", "--epoch",
        action="store", type=int
    )
    
    parser.add_argument(
        "-ea", "--epoch_als",
        action="store", type=int, default=500
    )
    
    parser.add_argument(
        "-lr", "--lr", action="store", type=float
    )        
    
    parser.add_argument(
        "-s", "--seed", 
        action="store", type=int,
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
            
    if args.action == "train_cp":
        start_time = time.time()
        _model = LightIT(_tensor, device, True, args)
        _model.quantization(args)     
        with open(args.output_path + ".txt", 'a') as f:
            f.write(f'cp time: {time.time() - start_time}\n')
            
    if args.action == "train":
        start_time = time.time()
        if os.path.exists(args.output_path + "_cp.pt"):            
            _model = LightIT(_tensor, device, False, args)                
            state_dict = torch.load(args.output_path + "_cp.pt", map_location=device)
            _model.centroids.data.copy_(state_dict['centroids'])

            for m in range(_tensor.order-2):
                _model.V[m].data.copy_(state_dict['V'][m].to(device))
            _model.S.data.copy_(state_dict['S'])
            _model.mapping = state_dict['mapping'].to(device)
            print(f"saved fitness: {state_dict['fitness']}")                        
        else:
            _model = LightIT(_tensor, device, True, args)
            _model.quantization(args)        
            clear_memory()
            with open(args.output_path + ".txt", 'a') as f:
                f.write(f'cp time: {time.time() - start_time}\n')
            start_time = time.time()    
        _model.als(args)        
        
        with open(args.output_path + ".txt", 'a') as f:
            f.write(f'tucker time: {time.time() - start_time}\n')