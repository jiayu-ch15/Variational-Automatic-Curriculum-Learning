#!/usr/bin/env python
import copy
import glob
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import MPEEnv
from algorithm.ppo import PPO,PPO3
from algorithm.autocurriculum import goal_proposal, make_parallel_env, evaluation, collect_data, collect_data_and_states, save
from algorithm.model import Policy,Policy3,ATTBase,ATTBase_actor_dist_add, ATTBase_critic_add, ATTBase_critic_add_littleinit

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage, RolloutStorage_share
from utils.single_storage import SingleRolloutStorage
import shutil
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import random
import copy
import matplotlib.pyplot as plt
import pdb
import wandb
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=10000)

# def trainer(envs, actor_critic, args):

def main():
    args = get_config()
    run = wandb.init(project='value_error_priority',name=str(args.algorithm_name) + "_seed" + str(args.seed))
    
    assert (args.share_policy == True and args.scenario_name == 'simple_speaker_listener') == False, ("The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    # path
    model_dir = Path('./results') / args.env_name / args.scenario_name / args.algorithm_name
    node_dir = Path('./node') / args.env_name / args.scenario_name / args.algorithm_name
    curricula_dir = Path('./curricula') / args.env_name / args.scenario_name / args.algorithm_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    if not node_dir.exists():
        node_curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in node_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            node_curr_run = 'run1'
        else:
            node_curr_run = 'run%i' % (max(exst_run_nums) + 1)
    curricula_curr_run = 'run%i'%args.seed

    run_dir = model_dir / curr_run
    save_node_dir = node_dir / node_curr_run
    save_curricula_dir = curricula_dir / curricula_curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    logger = SummaryWriter(str(log_dir)) 

    # initial hyper-parameters
    num_agents = args.num_agents
    buffer_length = 2000 # archive 长度
    boundary = {'x':[-0.3,0.3],'y':[-0.3,0.3]}
    start_boundary = {'x':[-0.3,0.3],'y':[-0.3,0.3]}
    restart_p = 0.3
    eval_frequency = 1 #需要fix几个回合
    check_frequency = 1
    use_gae = True
    use_one_step = True
    use_easy_sampling = False
    use_adaptive = False
    use_start_states = True
    use_states_clip = False
    use_little_init = True
    use_double_check = False
    save_node_fre = 5
    save_node = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    goals = goal_proposal(
        num_agents=num_agents, boundary=boundary, env_name=args.scenario_name, critic_k=1, 
        buffer_capacity=buffer_length, proposal_batch=args.n_rollout_threads, restart_p=restart_p, 
        score_type='value_error'
    )
    load_curricula = False
    warm_up = False
    only_eval = False

    # env
    envs = make_parallel_env(args)
    num_agents = args.num_agents
    #Policy network
    if args.share_policy:
        actor_base = ATTBase_actor_dist_add(envs.observation_space[0].shape[0], envs.action_space[0], num_agents)
        if use_little_init:
            critic_base = ATTBase_critic_add_littleinit(envs.observation_space[0].shape[0], num_agents)
        else:
            critic_base = ATTBase_critic_add(envs.observation_space[0].shape[0], num_agents)
        actor_critic = Policy3(envs.observation_space[0], 
                    envs.action_space[0],
                    num_agents = num_agents,
                    base=None,
                    actor_base=actor_base,
                    critic_base=critic_base,
                    base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                'recurrent': args.recurrent_policy,
                                'hidden_size': args.hidden_size,
                                'attn': args.attn,                                 
                                'attn_size': args.attn_size,
                                'attn_N': args.attn_N,
                                'attn_heads': args.attn_heads,
                                'dropout': args.dropout,
                                'use_average_pool': args.use_average_pool,
                                'use_common_layer':args.use_common_layer,
                                'use_feature_normlization':args.use_feature_normlization,
                                'use_feature_popart':args.use_feature_popart,
                                'use_orthogonal':args.use_orthogonal,
                                'layer_N':args.layer_N,
                                'use_ReLU':args.use_ReLU
                                },
                    device = device)
        actor_critic.to(device)
        # algorithm
        agents = PPO3(actor_critic,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss=args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   use_accumulate_grad=args.use_accumulate_grad,
                   use_grad_average=args.use_grad_average,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   device=device) 
        if load_curricula:
            agents.load(load_model_path=load_model_path, initial_optimizer=initial_optimizer)     
    else:
        actor_critic = []
        agents = []
        rollouts = []
        for agent_id in range(num_agents):
            ac = Policy(envs.observation_space, 
                      envs.action_space[agent_id],
                      num_agents = agent_id, # here is special
                      base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size,
                                 'attn': args.attn,                                 
                                 'attn_size': args.attn_size,
                                 'attn_N': args.attn_N,
                                 'attn_heads': args.attn_heads,
                                 'dropout': args.dropout,
                                 'use_average_pool': args.use_average_pool,
                                 'use_common_layer':args.use_common_layer,
                                 'use_feature_normlization':args.use_feature_normlization,
                                 'use_feature_popart':args.use_feature_popart,
                                 'use_orthogonal':args.use_orthogonal,
                                 'layer_N':args.layer_N,
                                 'use_ReLU':args.use_ReLU
                                 },
                      device = device)
            ac.to(device)
            # algorithm
            agent = PPO(ac,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   device=device)
                               
            actor_critic.append(ac)
            agents.append(agent)  
            #replay buffer
            ro = SingleRolloutStorage(agent_id,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space, 
                    envs.action_space,
                    args.hidden_size)
            rollouts.append(ro)

    # run
    begin = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    one_length_last = 0
    starts_length_last = 0
    one_length_now = args.n_rollout_threads
    starts_length_now = args.n_rollout_threads
    current_timestep = 0

    if use_adaptive:
        now_boundary = start_boundary
    else:
        now_boundary = boundary
    one_step = 0.1
    add_fre = 50
    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           

        if use_adaptive:
            if episode % add_fre == 0:
                now_boundary['x'] = [max(now_boundary['x'][0]- one_step,boundary['x'][0]), \
                                        min(now_boundary['x'][1]+ one_step,boundary['x'][1])]
                now_boundary['y'] = [max(now_boundary['y'][0]- one_step,boundary['y'][0]), \
                                    min(now_boundary['y'][1]+ one_step,boundary['y'][1])]
        wandb.log({'now_boundary':now_boundary['x'][1]},current_timestep)
        starts = goals.restart_sampling(
            boundary=now_boundary,
            start_boundary=start_boundary,
            args=args, 
            envs=envs, 
            agents=agents, 
            actor_critic=actor_critic, 
            use_gae=use_gae, 
            use_one_step=use_one_step,
            use_easy_sampling=use_easy_sampling,
            timestep=current_timestep
        )
        if episode % save_node_fre == 0 and save_node:
            goals.save_node(starts, save_node_dir, episode)
        if not only_eval:
            for times in range(eval_frequency):
                rollouts, current_timestep, restart_states, restart_states_value = collect_data_and_states(envs, agents, agents.actor_critic, args, starts, len(starts), len(starts), current_timestep)
                if use_start_states:
                    goals.add_restart_states(restart_states[0:len(starts)],restart_states_value[0:len(starts)], args, envs, agents, actor_critic, use_gae, use_one_step, use_double_check, use_states_clip)
                else:
                    goals.add_restart_states(restart_states,restart_states_value, args, envs, agents, actor_critic, use_gae, use_one_step, use_double_check, use_states_clip)
                # update the network
                if args.share_policy:
                    agents.actor_critic.train()
                    value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(args.num_agents, rollouts, current_timestep, warm_up=warm_up)
                    wandb.log({'value_loss': value_loss},current_timestep)
                    rollouts.after_update()
                else: # 需要修改成同时update的版本
                    value_losses = []
                    action_losses = []
                    dist_entropies = [] 
                    for agent_id in range(num_agents):
                        agents.actor_critic[agent_id].train()
                        value_loss, action_loss, dist_entropy = agents[agent_id].update_single(agent_id, rollouts[agent_id])
                        value_losses.append(value_loss)
                        action_losses.append(action_loss)
                        dist_entropies.append(dist_entropy)
                        rew = []
                        for i in range(rollouts[agent_id].rewards.shape[1]):
                            rew.append(np.sum(rollouts[agent_id].rewards[:,i]))
                        wandb.log({'average_episode_reward': np.mean(rew)},
                            (episode+1) * args.episode_length * one_length*eval_frequency)
                        rollouts[agent_id].after_update()

        
        # region evaluation
        if episode % check_frequency==0:
            print('----------evaluation-------------')
            test_starts = goals.uniform_sampling(starts_length=500,boundary=boundary)
            # current
            eval_num_agents = args.num_agents
            mean_cover_rate_current, eval_episode_reward_current = evaluation(envs, agents.actor_critic, args, eval_num_agents, current_timestep, test_starts)
            print('current cover rate ' + str(eval_num_agents) + ': ',mean_cover_rate_current)
            wandb.log({str(eval_num_agents) + 'cover_rate': mean_cover_rate_current}, current_timestep)
            # wandb.log({str(eval_num_agents) + 'success_rate': mean_success_rate_current}, current_timestep)
            # wandb.log({str(eval_num_agents) + 'test_collision_num': collision_num_current}, current_timestep)
            wandb.log({str(eval_num_agents) + 'eval_episode_reward': eval_episode_reward_current}, current_timestep)
            # # target
            # if eval_num_agents != args.eval_num_agents:
            #     eval_num_agents = args.eval_num_agents
            #     mean_cover_rate_target, mean_success_rate_target, collision_num_target, eval_episode_reward_target = evaluation(envs, actor_critic, args, eval_num_agents, current_timestep)
            #     print('target cover rate ' + str(eval_num_agents) + ': ',mean_cover_rate_target)
            #     wandb.log({str(eval_num_agents) + 'cover_rate': mean_cover_rate_target}, current_timestep)
            #     wandb.log({str(eval_num_agents) + 'success_rate': mean_success_rate_target}, current_timestep)
            #     wandb.log({str(eval_num_agents) + 'test_collision_num': collision_num_target}, current_timestep)
            #     wandb.log({str(eval_num_agents) + 'eval_episode_reward': eval_episode_reward_target}, current_timestep)
        # end region
        
        total_num_steps = current_timestep

        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({'model': agents.actor_critic}, str(save_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'model': agents.actor_critic[agent_id]
                                }, 
                                str(save_dir) + "/agent%i_model" % agent_id + ".pt")

        # log information
        if episode % args.log_interval == 0:
            end = time.time()
            print("\n Scenario {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(args.scenario_name,
                        args.algorithm_name,
                        episode, 
                        episodes,
                        total_num_steps,
                        args.num_env_steps,
                        int(total_num_steps / (end - begin))))
                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
if __name__ == "__main__":
    main()