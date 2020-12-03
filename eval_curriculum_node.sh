#!/bin/sh
ulimit -n 4096 
env="MPE"
scenario="simple_spread_3rooms_leftup_rightdown"
# scenario="push_ball"
num_landmarks=2
num_agents=2
algo="diversified_novelty_sp3_leftup_rightdown"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python eval_curriculum_node.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 2 --episode_length 70 --num_env_steps 30000000 --ppo_epoch 5 --recurrent_policy
    echo "training is done!"
done
