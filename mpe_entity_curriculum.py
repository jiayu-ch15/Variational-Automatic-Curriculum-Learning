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
from algorithm.autocurriculum import node_buffer, make_parallel_env, evaluation, collect_data, save
from algorithm.model import Policy,Policy3,ATTBase,ATTBase_actor_dist_add, ATTBase_critic_add

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
np.set_printoptions(linewidth=10000)

# def trainer(envs, actor_critic, args):

def main():
    args = get_config()
    run = wandb.init(project='entity_curriculum',name=str(args.algorithm_name) + "_seed" + str(args.seed))
    
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
    use_novelty_sample = True
    use_past_sampling = True
    del_switch = 'novelty'
    starts = []
    buffer_length = 2000 # archive 长度
    N_child = 325
    N_archive = 150
    N_parent = 25
    max_step = 0.6
    TB = 1
    M = N_child
    Rmin = 0.5
    Rmax = 0.95
    boundary = {'x':[-3,3],'y':[-3,3]}
    start_boundary = {'x':[-0.3,0.3],'y':[-0.3,0.3]}
    upper_bound = 0.9
    transfer = False
    if transfer:
        mix_flag = False
        decay = False
        stop_mix_signal = 1.0
        target_num = args.eval_num_agents
        last_box_num = 0
        last_agent_num = last_box_num
        now_box_num = 4
        now_agent_num = now_box_num
    else:
        mix_flag = False
        decay = True
        stop_mix_signal = 0.9
        mix_add_frequency = 30 # 改变比例的频率
        mix_add_count = 0
        ratio_current = 0.1
        if decay:
            ratio_last = (1.0 - ratio_current) * 2
        target_num = args.eval_num_agents
        last_box_num = 0
        last_agent_num = last_box_num
        now_box_num = 4
        now_agent_num = now_box_num
    last_mean_cover_rate = 0
    now_mean_cover_rate = 0
    eval_frequency = 3 #需要fix几个回合
    check_frequency = 1
    save_node_frequency = 3
    save_node_flag = False
    save_curricula = True
    next_stage_flag = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    last_node = node_buffer(last_agent_num,buffer_length,
                           archive_initial_length=int(args.n_rollout_threads),
                           reproduction_num=M,
                           max_step=max_step,
                           start_boundary=start_boundary,
                           boundary=boundary,
                           env_name=args.scenario_name)
    now_node = node_buffer(now_agent_num,buffer_length,
                           archive_initial_length=int(args.n_rollout_threads),
                           reproduction_num=M,
                           max_step=max_step,
                           start_boundary=start_boundary,
                           boundary=boundary,
                           env_name=args.scenario_name)

    # region load curricula and model
    load_curricula = True
    initial_optimizer = False
    warm_up = False
    warmup_iter = 150
    load_model_path = None
    if load_curricula: # 默认从4、8混合开始训练
        # load path
        load_curricula_path = Path('./curricula') / args.env_name / args.scenario_name / args.load_algorithm_name
        load_model_path = Path('./results') / args.env_name / args.scenario_name / args.load_algorithm_name 
        seed_path = 'run%i'%args.seed
        model_path = 'models/%iagent_model_0.9.pt'%args.load_num_agents
        load_curricula_path = load_curricula_path / seed_path / Path('%iagents'%args.load_num_agents)
        load_model_path = load_model_path / seed_path / model_path
        last_agent_num = 4
        now_agent_num = 8
        start_boundary = {'x':[-1,1],'y':[-1,1]}
        if not transfer:
            mix_flag = True
        # initialize now node
        now_node = node_buffer(now_agent_num,buffer_length,
                        archive_initial_length=int(args.n_rollout_threads),
                        reproduction_num=M,
                        max_step=max_step,
                        start_boundary=start_boundary,
                        boundary=boundary,
                        env_name=args.scenario_name)
        # load last node
        # load archive
        last_node.num_agents = last_agent_num
        with open(load_curricula_path / 'archive/archive_0.900000','r') as fp :
            tmp = fp.readlines()
            for i in range(len(tmp)):
                tmp[i] = np.array(tmp[i][1:-2].split(),dtype=float)
        archive_load = []
        for i in range(len(tmp)): 
            archive_load_one = []
            for j in range(last_node.num_agents * 2):
                archive_load_one.append(tmp[i][j*2:(j+1)*2])
            archive_load.append(archive_load_one)
        last_node.archive = copy.deepcopy(archive_load)
        # load parent
        with open(load_curricula_path / 'parent/parent_0.900000','r') as fp :
            tmp = fp.readlines()
            for i in range(len(tmp)):
                tmp[i] = np.array(tmp[i][1:-2].split(),dtype=float)
        parent_load = []
        for i in range(len(tmp)): 
            parent_load_one = []
            for j in range(last_node.num_agents * 2):
                parent_load_one.append(tmp[i][j*2:(j+1)*2])
            parent_load.append(parent_load_one)
        last_node.parent = copy.deepcopy(parent_load)
        # load parent_all
        with open(load_curricula_path / 'parent_all/parent_all_0.900000','r') as fp :
            tmp = fp.readlines()
            for i in range(len(tmp)):
                tmp[i] = np.array(tmp[i][1:-2].split(),dtype=float)
        parent_all_load = []
        for i in range(len(tmp)): 
            parent_all_load_one = []
            for j in range(last_node.num_agents * 2):
                parent_all_load_one.append(tmp[i][j*2:(j+1)*2])
            parent_all_load.append(parent_all_load_one)
        last_node.parent_all = copy.deepcopy(parent_all_load)
    # end region
    only_eval = False

    # env
    envs = make_parallel_env(args)
    num_agents = args.num_agents
    #Policy network
    if args.share_policy:
        actor_base = ATTBase_actor_dist_add(envs.observation_space[0].shape[0], envs.action_space[0], num_agents)
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
    warmup_count = 0
    entity_count = 0

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           

        # ratio
        if transfer:
            ratio_current = 1.0
            ratio_last = 0.0
        else:
            if decay: 
                if mix_add_count == mix_add_frequency and mix_flag:
                    mix_add_count = 0
                    ratio_current += 0.1
                    ratio_current = min(ratio_current,1.0)
                    ratio_last = (1.0 - ratio_current) * 2
                    # turn off mix
                    if np.abs(ratio_current - 1.0) <= 1e-5:
                        mix_flag = False
            else: # mixed switch
                ratio_last = (1.0 - ratio_current) * 2
        wandb.log({'ratio_current': ratio_current},current_timestep)
        
        # reproduction
        if mix_flag:
            last_node.reproduction_num = round(N_child * ratio_last)
            now_node.reproduction_num = round(N_child * ratio_current)
        else:
            last_node.reproduction_num = N_child
            now_node.reproduction_num = N_child
        if use_novelty_sample:
            if last_node.num_agents!=0:
                last_node.childlist += last_node.SampleNearby_novelty(last_node.parent, logger, current_timestep)
            now_node.childlist += now_node.SampleNearby_novelty(now_node.parent, logger, current_timestep)
        else:
            if last_node.num_agents!=0:
                last_node.childlist += last_node.SampleNearby(last_node.parent)
            now_node.childlist += now_node.SampleNearby(now_node.parent)
        
        # reset env 
        if transfer:
            if use_past_sampling:
                starts_now, one_length_now, starts_length_now = now_node.sample_starts(N_child,N_archive,N_parent)
            else:
                starts_now, one_length_now, starts_length_now = now_node.sample_starts(N_child,N_archive) 
            now_node.eval_score = np.zeros(shape=one_length_now) 
        else:
            # last node
            if last_node.num_agents!=0 and mix_flag:
                if use_past_sampling:
                    starts_last, one_length_last, starts_length_last = last_node.sample_starts(round(N_child*ratio_last),round(N_archive*ratio_last),round(N_parent*ratio_last))
                else:
                    starts_last, one_length_last, starts_length_last = last_node.sample_starts(round(N_child*ratio_last),round(N_archive*ratio_last))
                last_node.eval_score = np.zeros(shape=one_length_last)
            # now node
            if mix_flag:
                if use_past_sampling:
                    starts_now, one_length_now, starts_length_now = now_node.sample_starts(round(N_child*ratio_current),round(N_archive*ratio_current),round(N_parent*ratio_current))
                else:
                    starts_now, one_length_now, starts_length_now = now_node.sample_starts(round(N_child*ratio_current),round(N_archive*ratio_current))
            else:
                if use_past_sampling:
                    starts_now, one_length_now, starts_length_now = now_node.sample_starts(N_child,N_archive,N_parent)
                else:
                    starts_now, one_length_now, starts_length_now = now_node.sample_starts(N_child,N_archive)
            now_node.eval_score = np.zeros(shape=one_length_now)

        if not only_eval:
            for times in range(eval_frequency):
                if transfer:
                    # now node
                    rollouts_now, current_timestep = collect_data(envs, agents, agents.actor_critic, args, now_node, starts_now, starts_length_now, one_length_now, current_timestep)
                else:
                    if decay:
                        if mix_flag: mix_add_count += 1
                    # last node
                    if last_node.num_agents!=0 and mix_flag:
                        rollouts_last, current_timestep = collect_data(envs, agents, agents.actor_critic, args, last_node, starts_last, starts_length_last, one_length_last, current_timestep)
                    # now node
                    rollouts_now, current_timestep = collect_data(envs, agents, agents.actor_critic, args, now_node, starts_now, starts_length_now, one_length_now, current_timestep)

                # update the network
                if args.share_policy:
                    agents.actor_critic.train()
                    if warmup_count >= warmup_iter:
                        warm_up = False
                    if transfer:
                        wandb.log({'Type of agents': 1}, current_timestep)
                        value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(now_node.num_agents, rollouts_now, current_timestep, warm_up=warm_up)
                    else:
                        if last_node.num_agents!=0 and mix_flag:
                            wandb.log({'Type of agents': 2}, current_timestep)
                            value_loss, action_loss, dist_entropy = agents.update_double_share(last_node.num_agents, now_node.num_agents, rollouts_last, rollouts_now, current_timestep)
                            # clean the buffer and reset
                            rollouts_last.after_update()
                        else:
                            wandb.log({'Type of agents': 1},current_timestep)
                            value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(now_node.num_agents, rollouts_now, current_timestep, warm_up=False)
                    wandb.log({'value_loss': value_loss},current_timestep)
                    warmup_count += 1
                    rollouts_now.after_update()
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

            # move nodes
            if last_node.num_agents!=0 and mix_flag:
                last_node.eval_score = last_node.eval_score / eval_frequency
                last_node.move_nodes(one_length_last, Rmax, Rmin, del_switch, logger, current_timestep)
            now_node.eval_score = now_node.eval_score / eval_frequency
            now_node.move_nodes(one_length_now, Rmax, Rmin, del_switch, logger, current_timestep)
            # save node
            if (episode+1) % save_node_frequency ==0 and save_node_flag:
                last_node.save_node(save_node_dir, episode)
                now_node.save_node(save_node_dir, episode)
        
        # region evaluation
        if episode % check_frequency==0:
            print('----------evaluation-------------')
            # current
            eval_num_agents = now_node.num_agents
            mean_cover_rate_current, mean_success_rate_current, collision_num_current, eval_episode_reward_current = evaluation(envs, agents.actor_critic, args, eval_num_agents, current_timestep)
            print('current cover rate ' + str(eval_num_agents) + ': ',mean_cover_rate_current)
            wandb.log({str(eval_num_agents) + 'cover_rate': mean_cover_rate_current}, current_timestep)
            wandb.log({str(eval_num_agents) + 'success_rate': mean_success_rate_current}, current_timestep)
            wandb.log({str(eval_num_agents) + 'test_collision_num': collision_num_current}, current_timestep)
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
        
        # region entity curriculum
        if mean_cover_rate_current >= upper_bound and now_node.num_agents < target_num:
            mean_cover_rate_current = 0
            last_agent_num = now_node.num_agents
            now_agent_num = min(last_agent_num * 2,target_num)
            add_num = now_agent_num - last_agent_num
            if add_num!=0:
                if args.share_policy:
                    # save model when switch
                    save(str(save_dir) + "/%iagent_model_"%now_node.num_agents + str(upper_bound) + '.pt' , agents)
                    mix_flag = True
                    last_node = copy.deepcopy(now_node)
                    if save_curricula:
                        last_node.save_phase_curricula(save_curricula_dir, upper_bound)
                    start_boundary = 1.0
                    now_node = node_buffer(now_agent_num,buffer_length,
                                archive_initial_length=args.n_rollout_threads,
                                reproduction_num=M,
                                max_step=max_step,
                                start_boundary=start_boundary,
                                boundary=boundary,
                                env_name=args.scenario_name)
                    agents.actor_critic.num_agents = now_node.num_agents
                    if now_node.num_agents==8:
                        agents.num_mini_batch = 16
        # end region
                        
        # region turn off mixed_switch
        if not transfer:
            if not decay:
                if mean_cover_rate_current > stop_mix_signal:
                    print('---Turn off mixed-switch---')
                    mix_flag = False
                    decay_last = 0.0
                    decay_now = 1.0
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