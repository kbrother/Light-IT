python main.py -tp ../input/23-Irregular-Tensor/cms.npy -fp parafac2/cms_factor.mat -op results/cms.txt -de 0 -e 500 -lr 0.01 &
python main.py -tp ../input/23-Irregular-Tensor/ml_sample.npy -fp parafac2/ml_sample_factor.mat -op results/ml_sample.txt -de 1 -e 500 -lr 0.01 &
wait
