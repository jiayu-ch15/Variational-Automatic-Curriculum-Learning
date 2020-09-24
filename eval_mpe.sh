#!/bin/sh
env="MPE"
scenario_name="simple_spread"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=3 python eval_mpe.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --episode_length 70 --eval_episodes 50 --recurrent_policy --num_landmarks 2 --num_agents 2 --model_dir "/home/chenjy/mappo-sc/results/MPE/simple_spread/stage95_warmup_3iter/" --save_gifs
done
