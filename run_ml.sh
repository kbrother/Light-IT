for lr in 0.1 0.001
do
    for rank in 5 10 20 30
    do
        python main.py -tp ../data/23-Irregular-Tensor/ml-1m.npy -op results/ml-1m-lr${lr}-rank${rank} -de 5 -e 500 -lr ${lr} -r ${rank}        
    done
done
#python main.py -tp ../data/23-Irregular-Tensor/ml-1m.npy -fp parafac2/ml-1m-factor-10.mat -op results/ml-1m-rank10-parafac2 -de 7 -e 500 -lr 0.01 -r 10