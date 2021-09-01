#!/bin/sh
env="MPE"
scenario_name="simple_speaker_listener"
# scenario_name="push_ball"
seed=1

echo "env is ${env}"
CUDA_VISIBLE_DEVICES=0 python eval_mpe.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --episode_length 70 --eval_episodes 5 --recurrent_policy --num_landmarks 4 --num_agents 4 --model_dir "/home/tsing73/curriculum/results/MPE/simple_spread/valueerror_map2_p1.0_startscale1.0/" --share_policy # --save_gifs
