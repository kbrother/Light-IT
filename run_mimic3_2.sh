for rank in 10 20 30
do    
    python main.py -tp ../data/23-Irregular-Tensor/mimic3.npy -op results/mimic3-lr1-rank${rank} -de 3 -e 300 -lr 1 -r ${rank} & 
    python main.py -tp ../data/23-Irregular-Tensor/mimic3.npy -op results/mimic3-lr0.1-rank${rank} -de 4 -e 300 -lr 0.1 -r ${rank} & 
    python main.py -tp ../data/23-Irregular-Tensor/mimic3.npy -op results/mimic3-lr0.01-rank${rank} -de 5 -e 300 -lr 0.01 -r ${rank} & 
    wait
done