#!/bin/sh
env="BoxLocking"
scenario_name="quadrant"
num_agents=2
num_boxes=2
floor_size=12.0
task_type='all'
algo="pc_all_suc_warmup_floor12"
seed_max=1

ulimit -n 4096
export OPENBLAS_NUM_THREADS=1

echo "env is ${env}, scenario is ${scenario_name}, num_agents is ${num_agents}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train_boxlocking_pc.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --task_type ${task_type} --seed ${seed} --floor_size ${floor_size}  --n_rollout_threads 26 --num_mini_batch 2 --episode_length 120 --num_env_steps 100000000 --ppo_epoch 15 --attn --save_interval 1 --eval
    echo "training is done!"
done
