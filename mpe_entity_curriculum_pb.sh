#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="push_ball"
num_landmarks=2
num_agents=2
eval_num_agents=4
# algo='decay_fre300_pb_minibatch8_woclip'
# algo="transfer_pb_woclip_onlytrain4"
algo='check'
load_algo='transfer_pb_woclip'
load_num_agents=2
seed=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python mpe_entity_curriculum_pb.py --env_name ${env} --algorithm_name ${algo} --load_algorithm_name ${load_algo} --scenario_name ${scenario} --num_agents ${num_agents} --load_num_agents ${load_num_agents} --eval_num_agents ${eval_num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 8 --episode_length 120 --test_episode_length 200 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart --lr 5e-4 --use_accumulate_grad --max-grad-norm 20 # --use-max-grad-norm
