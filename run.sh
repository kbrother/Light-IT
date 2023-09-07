#python main.py -tp ../data/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor.mat -op results/cms-rank5 -de 3 -e 500 -lr 0.01 &
#python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor-10.mat -op results/cms-rank10 -de 0 -e 500 -lr 0.01 & 
#python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor-20.mat -op results/cms-rank20 -de 1 -e 500 -lr 0.01 &
python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor-40.mat -op results/cms-rank40 -de 2 -e 500 -lr 0.01 &

python main.py -tp ../data/23-Irregular-Tensor/ml-1m.npy -fp parafac2/ml-1m-factor.mat -op results/ml-1m-rank5 -de 3 -e 500 -lr 0.01 &
python main.py -tp ../data/23-Irregular-Tensor/ml-1m.npy -fp parafac2/ml-1m-factor-10.mat -op results/ml-1m-rank10 -de 0 -e 500 -lr 0.01 &
python main.py -tp ../data/23-Irregular-Tensor/ml-1m.npy -fp parafac2/ml-1m-factor-20.mat -op results/ml-1m-rank20 -de 1 -e 500 -lr 0.01 &
#python main.py -tp ../data/23-Irregular-Tensor/ml-1m.npy -fp parafac2/ml-1m-factor-40.mat -op results/ml-1m-rank40 -de 3 -e 500 -lr 0.01 &
wait
