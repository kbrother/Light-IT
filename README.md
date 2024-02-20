# Compact Decomposition of Irregular Tensors for Data Compression: From Sparse to Dense to High-Order Tensors

This repository is the official implementation of Compact Decomposition of Irregular Tensors for Data Compression: From Sparse to Dense to High-Order Tensors.

## Requirements
Please see the requirements.txt
```
numpy==1.21.6
scipy==1.7.3
torch==1.13.1
tqdm==4.51.0
```

## Running Light-IT and Light-IT++
The training processes of Light-IT and Light-IT++ are implemented in ```main.py```.
### Positional argument
* `action`: `train_cp` when running only Light-IT, `train` when running Light-IT and Light-IT++.
* `-tp`, `--tensor_path`:  file path for an irregular tensor. A file should be pickle files('.pickle') for sparse tensors and numpy files ('.numpy') for dense tensors.
* `-op`, `--output_path`: output path for saving the parameters and fitness.
* `-r`, `--rank`: rank of the model.
* `-d`, `--is_dense`: 'True' when the input tensor is dense, 'False' when the input tensor is sparse.
* `-e`, `--epoch`: Number of epochs for Light-IT.
* `-lr`, `--lr`: Learning rate for Light-IT.
* `-s`, `--seed`: Seed of execution.

**Please reduce the batch sizes (-bz, -bnz, -cb, -tbz, -tbnz, and -tbnx) when O.O.M occurs in GPU!**
### Optional arguments (common)
* `-de`, `--device`: GPU id for execution.
* `-bz`, `--batch_lossz`: Batch size for computing the loss (corresponding to the zero entries) of Light-IT.
* `-bnz`, `--batch_lossnz`: Batch size for computing the loss (corresponding to the non-zero entries) of Light-IT.
* `-cb`, `--cb`: Batch size for the clustering process of Light-IT.

### Optional arguments (Light-IT++)
* `-ea`, `--epoch_als`: Number of epochs for Light-IT++.
* `-tbz`, `--tucker_batch_lossz`: Batch size for the operations (dealing with zero entries) in ALS.
* `-tbnz`, `--tucker_batch_lossnz`: Batch size for the operations (dealing with non-zero entries) in ALS.
* `-tbnx`, `--tucker_batch_alsnx`: Batch size for the operations (not related to the tensor entries) in ALS.

### Example command 
```
  # Run Light-IT only
  python 23-Irregular-Tensor/main.py train_cp -tp ../input/23-Irregular-Tensor/usstock.npy -op output/usstock_r4_s0_lr0.01 -r 4 -d True -de 0 -lr 0.01 -e 500 -s 0

  # Run Light-IT and Light-IT++
  python 23-Irregular-Tensor/main.py train -tp ../input/23-Irregular-Tensor/usstock.npy -op output/usstock_r4_s0_lr0.01 -r 4 -d True -de 0 -lr 0.01 -e 500 -s 0 
```

