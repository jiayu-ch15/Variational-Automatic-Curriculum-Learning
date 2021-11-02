#!/bin/sh
ulimit -n 32768
env="MPE"
# scenario="push_ball"
scenario="hard_spread"
num_landmarks=4
num_agents=4
# algo="woparentsampling_sp_4agents"
# algo="check"
algo='VACL_hs_final'
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python train_mpe_woEP.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart --lr 5e-4 --use_accumulate_grad \
--buffer_length 2000 --epsilon 0.6 --delta 0.6 --h 1000 --archive_initial_length 500 --B_exp 300 --use_wandb

# CUDA_VISIBLE_DEVICES=0 python train_mpe_woEP.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 120 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart --lr 5e-4 --use_accumulate_grad \
# --buffer_length 2000 --epsilon 0.4 --delta 0.4 --h 1 --archive_initial_length 500 --B_exp 150 --use_wandb

# CUDA_VISIBLE_DEVICES=1 python train_mpe_woEP.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 60000000 --ppo_epoch 15 --recurrent_policy --use_popart --lr 5e-4 --use_accumulate_grad \
# --buffer_length 2000 --epsilon 0.6 --delta 0.6 --sol_prop 0.05 --use_wandb

