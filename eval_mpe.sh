#!/bin/sh
env="MPE"
scenario_name="simple_spread"
# scenario_name="push_ball"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python eval_mpe.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --episode_length 100 --eval_episodes 300 --recurrent_policy --num_landmarks 4 --num_agents 4 --model_dir "/home/tsing73/curriculum/results/MPE/simple_spread/valueerror_map2_p1.0_startscale1.0/" --save_gifs
done
