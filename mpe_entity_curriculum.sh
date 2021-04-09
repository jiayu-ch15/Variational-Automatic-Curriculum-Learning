#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="simple_spread"
num_landmarks=4
num_agents=4
eval_num_agents=8
# algo='mixed_ratio55_0.9_initial_optimizer'
# algo='transfer_0.9_initial_optimizer'
# algo='transfer_0.9_load_optimizer_check'
algo='decay_fre30iter_add0.1_load_optimizer'
# algo='transfer_warmup30iter'
# algo='transfer_switch0.9_check3'
# algo='check'
load_algo='transfer_switch0.9'
load_num_agents=4
seed=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=1 python mpe_entity_curriculum.py --env_name ${env} --algorithm_name ${algo} --load_algorithm_name ${load_algo} --scenario_name ${scenario} --num_agents ${num_agents} --load_num_agents ${load_num_agents} --eval_num_agents ${eval_num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 16 --episode_length 70 --test_episode_length 100 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart --lr 5e-4 --use_accumulate_grad
