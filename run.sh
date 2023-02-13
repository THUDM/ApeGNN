# AMiner
## HeatKernel
nohup python main.py --dataset aminer --gnn ApeGNN_HT --pool sum --Ks '[20, 50]' --step 1 --runs 1 --e 1e-7 --gpu_id 1 > ./logs/aminer/ApeGNN_HT.log 2>&1 &
## APPNP
nohup python main.py --dataset aminer --gnn ApeGNN_APPNP --pool sum --Ks '[20, 50]' --step 1 --runs 1 --e 1e-7 --gpu_id 1 > ./logs/aminer/ApeGNN_APPNP.log 2>&1 &
