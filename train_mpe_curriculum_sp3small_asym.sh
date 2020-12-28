#!/bin/sh
ulimit -n 32768
env="MPE"
scenario="simple_spread_3rooms_small_asym"
num_landmarks=4
num_agents=4
algo="tech3_sp3_small_asym"
# algo='check'
seed=2

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=3 python train_mpe_curriculum_sp3small_asym.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 150000000 --ppo_epoch 15 --recurrent_policy --use_popart
