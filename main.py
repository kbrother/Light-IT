import argparse
from parafac2 import parafac2
from data import irregular_tensor
import torch
import gc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', "--tensor_path", type=str)
    parser.add_argument('-op', "--output_path", type=str)
    parser.add_argument('-fp', "--factor_path", type=str)
    
    parser.add_argument(
        "-de", "--device",
        action="store", type=int
    )    
    
    parser.add_argument(
        "-b", "--batch_size",
        action="store", default=2**23, type=int
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
        "-tbz", "--tucker_batch_lossz",
        action="store", default=2**6, type=int
    )
    
    parser.add_argument(
        "-tbn", "--tucker_batch_lossnz",
        action="store", default=2**11, type=int
    )
    
    parser.add_argument(
        "-tbu", "--tucker_batch_u",
        action="store", default=2**6, type=int
    )
        
    parser.add_argument(
        "-tbv", "--tucker_batch_v",
        action="store", default=2**7, type=int
    )
    
    parser.add_argument(
        "-tbs", "--tucker_batch_s",
        action="store", default=2**7, type=int
    )
    
    parser.add_argument(
        "-tbg", "--tucker_batch_g",
        action="store", default=2**7, type=int
    )
    
    args = parser.parse_args()    
    device = torch.device("cuda:" + str(args.device))
    _tensor = irregular_tensor(args.tensor_path, device)
    print("load finish")
    
    _parafac2 = parafac2(_tensor, device, args)    
    _parafac2.quantization(args)
    gc.collect()
    torch.cuda.empty_cache()    
    _parafac2.als(args)