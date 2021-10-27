#!/bin/sh
ulimit -n 32768
env="MPE"
scenario="simple_speaker_listener_goal_conditioned"
num_landmarks=3
num_agents=2
algo="sl_amigo"
# algo='check'
seed=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python sl_amigo.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 100 --num_mini_batch 1 --episode_length 70 --episode_length_teacher 5 --teacher_lr 1e-3 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart --use_accumulate_grad --share_policy 
