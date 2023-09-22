for rank in 5
do
    lr_list=(0.1 0.01 0.001)
    device=(5 6 7)
    for i in 0 1 2
    do
        python main.py train -tp ../data/23-Irregular-Tensor/activity.npy -op results/activity-lr${lr_list[$i]}-rank${rank} -de ${device[$i]} -e 500 -lr ${lr_list[$i]} -r ${rank} -d True &            
    done        
    wait
done