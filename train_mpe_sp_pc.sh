#!/usr/bin/python
ulimit -n 8192
env="MPE"
scenario="simple_spread"
num_landmarks=4
num_agents=4
# algo='pc_sp'
algo='sp_4agents'
# algo='check'
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=1 python train_mpe_sp_pc.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 100 --num_mini_batch 1 --episode_length 70 --num_env_steps 40000000 --ppo_epoch 15  --recurrent_policy --use-max-grad-norm --use_popart
