#!/usr/bin/python
ulimit -n 8192
env="MPE"
scenario="simple_spread"
num_landmarks=4
num_agents=4
# algo='ours_warmup6iter'
algo='ours_withoutwarmup'
# algo='check'
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

# for seed in `seq ${seed_max}`;
# do
#     echo "seed is ${seed}:"
#     CUDA_VISIBLE_DEVICES=0 python train_mpe_curriculum_stage.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 16 --episode_length 70 --num_env_steps 40000000 --ppo_epoch 15 --entropy_coef 0.01  --recurrent_policy --use-max-grad-norm --use_popart
#     echo "training is done!"
# done
seed=1
CUDA_VISIBLE_DEVICES=0 python train_mpe_curriculum_stage.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 16 --episode_length 70 --num_env_steps 40000000 --ppo_epoch 15 --entropy_coef 0.01  --recurrent_policy --use-max-grad-norm --use_popart
