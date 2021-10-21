#!/bin/sh
ulimit -n 32768
env="MPE"
scenario="simple_spread"
# scenario="push_ball"
# scenario="hard_spread"
num_landmarks=4
num_agents=4
epsilon=0.6
delta=0.6
algo='sp3_4agents_cameraready'
# algo='pb_2agents_cameraready'
# algo='check'
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python sp_VACL.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --epsilon ${epsilon} --delta ${delta}  --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 80000000 --ppo_epoch 15 --use_popart --lr 5e-4 --use_accumulate_grad \
--recurrent_policy --use_wandb
