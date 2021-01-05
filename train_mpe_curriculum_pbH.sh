#!/bin/sh
ulimit -n 16384
env="MPE"
scenario="push_ball_H"
num_landmarks=2
num_agents=2
algo="tech1_pbH_left2right_hall1.0"
# algo="tech3_left2right_nohall"
# algo="check"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python train_mpe_curriculum_pbH.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 1 --episode_length 120 --num_env_steps 250000000 --ppo_epoch 15 --recurrent_policy --use_popart
