#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="push_ball"
num_landmarks=2
num_agents=2
num_boxes=2
algo='pb_2agents'
# algo='check'
seed=3

CUDA_VISIBLE_DEVICES=3 python train_mpe_pb_pc.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --num_boxes ${num_boxes}  --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 120 --num_env_steps 80000000 --ppo_epoch 15 --recurrent_policy --use_popart
