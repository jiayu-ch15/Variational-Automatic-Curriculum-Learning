#!/bin/sh
env="BoxLocking"
scenario_name="empty"
num_agents=2
num_boxes=2
floor_size=12.0
grid_size=60
task_type='all-return'
algo='bl_wo_evaluation_step4'
#algo="check"
seed=2

ulimit -n 4096
export OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python bl_wo_evaluation.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --task_type ${task_type} --seed ${seed} --floor_size ${floor_size} --grid_size ${grid_size}  --n_rollout_threads 300 --episode_length 60 --env_horizon 60 --num_mini_batch 1  --num_env_steps 170000000 --ppo_epoch 15 --attn --save_interval 50 --eval --spwan_obs --data_chunk_length 40
