#!/bin/sh
env="BoxLocking"
scenario_name="empty"
num_agents=2
num_boxes=2
floor_size=12.0
grid_size=60
task_type='all-return'
algo='debug'
seed=1

CUDA_VISIBLE_DEVICES=1 python train_bl.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --task_type ${task_type} --seed ${seed} --floor_size ${floor_size} --grid_size ${grid_size}  --n_rollout_threads 300 --episode_length 60 --env_horizon 60 --num_mini_batch 1  --num_env_steps 200000000 --ppo_epoch 15 --attn --save_interval 50 --eval --spawn_obs --data_chunk_length 40 \
--buffer_length 2000 --epsilon 6 --delta 6 --sol_prop 0.05 --B_exp 200 --h 1 --archive_initial_length 500 --eval_number 3 --use_wandb
