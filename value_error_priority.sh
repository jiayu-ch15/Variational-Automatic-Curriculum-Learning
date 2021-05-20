#!/bin/sh
# ulimit -n 32768
env="MPE"
scenario="simple_spread"
num_landmarks=4
num_agents=4
algo="valueerror_map2_p0.9_boundary0.3_onlystart_littleinit_true"
# algo="valueerror_map2_p1.0_startscale1.0"
# algo='check'
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python value_error_priority.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --test_episode_length 70  --num_env_steps 60000000 --ppo_epoch 15 --recurrent_policy --use_popart --use-max-grad-norm
