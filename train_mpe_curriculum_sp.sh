#!/bin/sh
ulimit -n 16384
env="MPE"
scenario="simple_spread"
num_landmarks=4
num_agents=4
algo="ours_1_3"
seed_max=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train_mpe_curriculum_sp.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 120 --num_env_steps 60000000 --ppo_epoch 15 --recurrent_policy --use_popart
    echo "training is done!"
done
