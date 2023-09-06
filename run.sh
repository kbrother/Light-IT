python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor.mat -op results/cms-rank5.txt -de 3 -e 500 -lr 0.01 &
python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor-10.mat -op results/test.txt -de 0 -e 20 -lr 0.01 & 
python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor-20.mat -op results/cms-rank20.txt -de 1 -e 500 -lr 0.01 &
python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor-40.mat -op results/cms-rank40.txt -de 2 -e 500 -lr 0.01 &
#python main.py -tp ../input/23-Irregular-Tensor/ml-1m.npy -fp parafac2/ml-1m-factor.mat -op results/ml-1m-gd.txt -de 1 -e 500 -lr 0.01 &
wait
