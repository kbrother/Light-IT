python main.py train -tp ../data/23-Irregular-Tensor/action.npy -op results/action-lr0.001-rank40 -de 0 -e 500 -lr 0.001 -r 40 -d True &
python main.py train -tp ../data/23-Irregular-Tensor/action.npy -op results/action-lr0.1-rank40 -de 2 -e 500 -lr 0.1 -r 40 -d True &
python main.py train -tp ../data/23-Irregular-Tensor/action.npy -op results/action-lr0.01-rank40 -de 3 -e 500 -lr 0.01 -r 40 -d True &
wait