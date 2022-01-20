#!/bin/sh
env="hide_and_seek"
scenario_name="quadrant"
num_seekers=1
num_hiders=1
num_boxes=1
num_ramps=1
num_food=0
floor_size=6.0
grid_size=30
seed=1
model_dir='/Users/chenjy/Desktop/VACL/results/hideandseek'

echo "env is ${env}"
CUDA_VISIBLE_DEVICES=0 python render_hns.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --num_seekers ${num_seekers} --num_hiders ${num_hiders} --num_boxes ${num_boxes} --num_ramps ${num_ramps} --num_food ${num_food} \
--use_render --model_dir ${model_dir}
