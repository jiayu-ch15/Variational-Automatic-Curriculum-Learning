#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="simple_spread"
num_landmarks=4
num_agents=4
algo='mixdecay4n8_adjustratio_util0.9'
# algo='check'
seed=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python train_mpe_mix_decay_followlast_sp.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 1000 --num_mini_batch 8 --episode_length 70 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart 
