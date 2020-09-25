#!/bin/sh
env="MPE"
# scenario_name="simple_spread"
scenario_name="simple_spread"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=7 python eval_mpe.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --episode_length 300 --eval_episodes 50 --recurrent_policy --num_landmarks 8 --num_agents 8 --model_dir "/home/chenjy/mappo-sc/results/MPE/simple_spread/check/" --save_gifs
done
