#!/bin/sh
ulimit -n 32768
env="MPE"
scenario="simple_spread_3rooms"
num_landmarks=4
num_agents=4
# algo="check"
algo='tech2_sp3H'
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=1 python train_mpe_curriculum_sp3.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 80000000 --ppo_epoch 15 --recurrent_policy --use_popart
