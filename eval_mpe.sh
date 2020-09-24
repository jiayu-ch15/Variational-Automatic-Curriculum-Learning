#!/bin/sh
env="MPE"
# scenario_name="simple_spread"
scenario_name="push_ball"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=3 python eval_mpe.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --episode_length 10 --eval_episodes 50 --recurrent_policy --num_landmarks 2 --num_agents 2 --model_dir "/home/chenjy/mappo-sc/results/MPE/push_ball/check/" --save_gifs
done
