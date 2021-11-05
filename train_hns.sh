#!/bin/sh
env="hidenseek"
scenario_name="quadrant"
num_hiders=1
num_seekers=1
num_boxes=1
num_ramps=1
floor_size=6.0
task_type='all-return'
# algo='hns_h1_woPC_eval3_step3_wointclip_middlestep_init500_Bexp200'
algo='check'
seed=1

ulimit -n 4096

CUDA_VISIBLE_DEVICES=1 python train_hns.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_hiders ${num_hiders} --num_seekers ${num_seekers} --num_boxes ${num_boxes} --num_ramps ${num_ramps}  --task_type ${task_type} --seed ${seed} --floor_size ${floor_size}  --n_rollout_threads 2 --num_mini_batch 2 --episode_length 60 --env_horizon 60 --num_env_steps 400000000 --ppo_epoch 15 --attn --save_interval 1 --eval \
--buffer_length 2000 --epsilon 3 --delta 3 --sol_prop 0.05 --B_exp 200 --h 1 --archive_initial_length 500 --eval_number 3 
# --use_wandb
