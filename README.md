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
### Positional arguments
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

### Example output
* `usstock_r4_s0_lr0.01.txt`: Saved the running time and fitness
* `usstock_r4_s0_lr0.01_cp.pt`: Saved the parameters of Light-IT
* `usstock_r4_s0_lr0.01.pt`: Saved the parameters of Light-IT++

## Checking the compressed size of the parameters
Checking the compressed sizes of Light-IT and Light-IT++ are implemented in ```huffman.py```.
### Positional arguments
* `-tp, -r, -d, -de, -bz, -bnz, -cb, -tbz, -tbnz, -tbnx`: same with the cases of running Light-IT and Light-IT++.
* `-rp`, `--result_path`: path for the '.pt' file.
* `-cp`, `--is_cp`: "True" when using the output of Light-IT, "False" when using the output of Light-IT++
### Example command
```
python huffman.py -tp ../data/23-Irregular-Tensor/cms.pickle -rp results/cms-lr0.01-rank5.pt -cp False -r 5 -de 0 -d False
```

## Real-world datasets which we used
|Name|N_max|N_avg|Size (except the 1st mode)|Order|Density|Source|Download Link|
|-|-|-|-|-|-|-|-|
|CMS|175|35.4|284 x 91,586|3|0.00501|[US government](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf)|[Link](https://www.dropbox.com/scl/fi/v08po2cqscefhd4gxa0qa/cms.pickle?rlkey=a0dk7mval7s3n1cetpuotjwge&dl=0)| 
|MIMIC-III|280|12.3|1,000 x 37,163|3|0.00733|[MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)|[Link](https://www.dropbox.com/scl/fi/m306thilnzdbv9m76dgvk/mimic3.pickle?rlkey=em9mbyh81sqzp3dnhdave8ry9&dl=0)|
|Korea-stock|5,270|3696.5|88 x 1,000|3|0.998|[DPar2](https://datalab.snu.ac.kr/dpar2/)|[Link](https://www.dropbox.com/scl/fi/kvnhu9pst84230cb86qmg/kstock.npy?rlkey=nmk7v3n4s2gztrbizxdjxk2oo&dl=0)|
|US-stock|7,883|3912.6|88 x 1,000|3|1|[DPar2](https://datalab.snu.ac.kr/dpar2/)|[Link](https://www.dropbox.com/scl/fi/opmlfm2u7808hwhrjxzi4/usstock.npy?rlkey=jm61ntlcj0o78cupvwkyg5z96&dl=0)|
|Enron|554|80.6|1,000 x 1,000 x 939|4|0.0000693|[FROSTT](https://frostt.io/tensors/enron/)|[Link](https://www.dropbox.com/scl/fi/v3und62rvn90c37yeknr8/enron.pickle?rlkey=4i6derahcvl3xfl0mdiadv4pj&dl=0)|
|Delicious|312|16.4|1,000 x 1,000 x 31,311|4|0.00000397|[FROSTT](https://frostt.io/tensors/delicious/)|[Link](https://www.dropbox.com/scl/fi/9krclnckqh09qp0fmtun2/delicious.pickle?rlkey=t4t87oqqexclqoun69n5lsdek&dl=0)|
