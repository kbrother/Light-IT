for rank in 40
do    
    python main.py -tp ../data/23-Irregular-Tensor/mimic3.npy -op results/mimic3-lr1-rank${rank} -de 0 -e 300 -lr 1 -r ${rank} & 
    python main.py -tp ../data/23-Irregular-Tensor/mimic3.npy -op results/mimic3-lr0.1-rank${rank} -de 1 -e 300 -lr 0.1 -r ${rank} & 
    python main.py -tp ../data/23-Irregular-Tensor/mimic3.npy -op results/mimic3-lr0.01-rank${rank} -de 2 -e 300 -lr 0.01 -r ${rank} & 
    wait
done