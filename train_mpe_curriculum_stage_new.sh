#!/usr/bin/python
ulimit -n 4096 
env="MPE"
scenario="simple_spread_atoc"
num_landmarks=4
num_agents=4
algo="check"
# algo='fix_init_set_map44'
# algo='zhang'
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train_mpe_curriculum_stage_new.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 500 --num_mini_batch 512 --episode_length 150 --num_env_steps 30000000 --ppo_epoch 15 --entropy_coef 0.01  --recurrent_policy --use_popart --use-max-grad-norm
    echo "training is done!"
done
