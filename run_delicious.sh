python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r5_lr0.01 -r 5 -d False -de 4 -e 500 -lr 0.01 &
python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r5_lr0.001 -r 5 -d False -de 5 -e 500 -lr 0.001 &
wait

python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r10_lr0.01 -r 10 -d False -de 4 -e 500 -lr 0.01 &
python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r10_lr0.001 -r 10 -d False -de 5 -e 500 -lr 0.001 &
wait

python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r20_lr0.01 -r 20 -d False -de 4 -e 500 -lr 0.01 &
python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r20_lr0.001 -r 20 -d False -de 5 -e 500 -lr 0.001 &
wait

python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r30_lr0.01 -r 30 -d False -de 4 -e 500 -lr 0.01 &
python main.py train -tp ../data/23-Irregular-Tensor/delicious.pickle -op results/delicious_r30_lr0.001 -r 30 -d False -de 5 -e 500 -lr 0.001 &
wait