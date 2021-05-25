#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="simple_spread_3rooms_left2right"
# scenario="push_ball_H"
num_agents=4
num_landmarks=4
num_boxes=4
# algo="diversified_novelty_parentsampling_sp3_leftup_rightdown"
algo='sp3_gradient_sample'
seed=1

CUDA_VISIBLE_DEVICES=0 python eval_curriculum_node.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --num_boxes ${num_boxes} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 30000000 --ppo_epoch 5 --recurrent_policy
