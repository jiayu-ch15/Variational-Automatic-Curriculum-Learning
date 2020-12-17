#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="push_ball"
num_landmarks=2
num_agents=2
algo="mix2n4_mix_decay_pb_mixdecayfre30"
# algo="check"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

# for seed in `seq ${seed_max}`;
# do
#     echo "seed is ${seed}:"
#     CUDA_VISIBLE_DEVICES=3 python train_mpe_mix_pb.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 120 --num_env_steps 100000000 --ppo_epoch 15 --recurrent_policy --use_popart
#     echo "training is done!"
# done
seed=3
CUDA_VISIBLE_DEVICES=3 python train_mpe_mix_decay_pb.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 1 --episode_length 120 --num_env_steps 300000000 --ppo_epoch 15 --recurrent_policy --use_popart