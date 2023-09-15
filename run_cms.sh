#python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms-factor-40.mat -op results/cms-rank40 -de 2 -e 500 -lr 0.01 &

for lr in 0.1 0.01 0.001
do
    for rank in 5 10 20 30 40
    do
        python main.py -tp ../data/23-Irregular-Tensor/cms.npy -op results/cms-lr${lr}-rank${rank} -de 7 -e 500 -lr ${lr} -r ${rank}        
    done
done