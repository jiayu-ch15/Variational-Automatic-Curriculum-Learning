#!/bin/sh
env="MPE"
# scenario_name="simple_spread"
scenario_name="simple_spread"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=2 python eval_mpe.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --episode_length 200 --eval_episodes 50 --recurrent_policy --num_landmarks 50 --num_agents 50 --model_dir "/home/chenjy/mappo-sc/results/MPE/simple_spread/eval_add_obs/" # --save_gifs
done
