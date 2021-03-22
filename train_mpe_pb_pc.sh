#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="push_ball"
num_landmarks=4
num_agents=4
num_boxes=4
algo='pb_4agents'
# algo='check'
# seed_max=1

# echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

# for seed in `seq ${seed_max}`;
# do
#     echo "seed is ${seed}:"
#     CUDA_VISIBLE_DEVICES=1 python train_mpe_curriculum_pb_stage.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --num_boxes ${num_boxes}  --seed ${seed} --n_rollout_threads 500 --num_mini_batch 1 --episode_length 120 --num_env_steps 600000000 --ppo_epoch 15 --recurrent_policy --use_popart
#     echo "training is done!"
# done
seed=3
CUDA_VISIBLE_DEVICES=3 python train_mpe_pb_pc.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --num_boxes ${num_boxes}  --seed ${seed} --n_rollout_threads 500 --num_mini_batch 8 --episode_length 120 --num_env_steps 400000000 --ppo_epoch 15 --recurrent_policy --use_popart
