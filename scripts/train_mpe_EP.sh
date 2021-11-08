#!/bin/sh
ulimit -n 32768
env="MPE"
scenario="simple_spread"
num_landmarks=4
num_agents=4
algo="debug"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

# simple spread
CUDA_VISIBLE_DEVICES=0 python train_mpe_EP.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 60000000 --ppo_epoch 15 --recurrent_policy --use_popart --lr 5e-4 --use_accumulate_grad \
--buffer_length 2000 --epsilon 0.6 --delta 0.6 --h 1 --archive_initial_length 500 --B_exp 150 --threshold_next 0.9 --decay_interval 30 --num_target 8 --use_wandb

# push ball
# CUDA_VISIBLE_DEVICES=0 python train_mpe_EP.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 120 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart --lr 5e-4 --use_accumulate_grad \
# --buffer_length 2000 --epsilon 0.4 --delta 0.4 --h 1 --archive_initial_length 500 --B_exp 150 --decay_interval 300 --num_target 4 --use_wandb

