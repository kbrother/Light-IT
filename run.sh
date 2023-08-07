python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms_factor.mat -op results/cms-gd.txt -de 0 -e 500 -lr 0.01 &
python main.py -tp ../input/23-Irregular-Tensor/ml-1m.npy -fp parafac2/ml-1m-factor.mat -op results/ml-1m-gd.txt -de 1 -e 500 -lr 0.01 &
wait
