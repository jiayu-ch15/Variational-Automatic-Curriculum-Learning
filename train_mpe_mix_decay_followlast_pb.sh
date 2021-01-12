#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="push_ball"
num_landmarks=2
num_agents=2
algo="mixdecay2n4_adjustratio_util0.9"
# algo="check"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

seed=3
CUDA_VISIBLE_DEVICES=2 python train_mpe_mix_decay_followlast_pb.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 1000 --num_mini_batch 8 --episode_length 120 --num_env_steps 300000000 --ppo_epoch 15 --recurrent_policy --use_popart