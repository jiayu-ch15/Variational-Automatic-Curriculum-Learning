#!/bin/sh
env="hidenseek"
scenario_name="quadrant"
num_hiders=1
num_seekers=1
num_boxes=1
num_ramps=1
floor_size=6.0
task_type='all-return'
# algo="Hidenseek_2agent_1box_1ramp_floor6_env300_step60_minibatch2"
algo='hns_h1_woPC_eval1_step1_wointclip_middlestep_init500_Bexp100'
seed=1

ulimit -n 4096

CUDA_VISIBLE_DEVICES=2 python train_hns.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_hiders ${num_hiders} --num_seekers ${num_seekers} --num_boxes ${num_boxes} --num_ramps ${num_ramps}  --task_type ${task_type} --seed ${seed} --floor_size ${floor_size}  --n_rollout_threads 300 --num_mini_batch 2 --episode_length 60 --env_horizon 60 --num_env_steps 400000000 --ppo_epoch 15 --attn --save_interval 1 --eval \
--buffer_length 2000 --epsilon 1 --delta 1 --sol_prop 0.05 --B_exp 100 --h 1 --archive_initial_length 500 --use_wandb
