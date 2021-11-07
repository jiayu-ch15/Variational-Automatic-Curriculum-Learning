#!/usr/bin/env python
import sys
sys.path.append("..")
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

from algorithm.autocurriculum import node_buffer, log_infos, make_parallel_env_mpe, evaluation, collect_data, save
from algorithm.ppo import PPO
from algorithm.model import Policy, ATTBase_actor, ATTBase_critic

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
from utils.single_storage import SingleRolloutStorage
import shutil
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import random
import copy
import wandb
import warnings
np.set_printoptions(linewidth=1000)
warnings.filterwarnings('ignore')

def main():
    args = get_config()
    if args.use_wandb:
        run = wandb.init(project=args.scenario_name,name=str(args.algorithm_name) + "_seed" + str(args.seed))
    
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

    run_dir = model_dir / curr_run
    save_node_dir = node_dir / node_curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    logger = SummaryWriter(str(log_dir)) 

    # env
    envs = make_parallel_env_mpe(args)
    num_agents = args.num_agents
    #Policy network
    if args.share_policy:
        model_name = args.scenario_name
        actor_base = ATTBase_actor(envs.observation_space[0].shape[0], envs.action_space[0], num_agents, model_name)
        critic_base = ATTBase_critic(envs.observation_space[0].shape[0], num_agents, model_name)
        actor_critic = Policy(envs.observation_space[0], 
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
        agents = PPO(actor_critic,
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
                   use_accumulate_grad=args.use_accumulate_grad,
                   use_grad_average=args.use_grad_average,
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
    
    starts = []
    N_parent = int(args.n_rollout_threads * args.sol_prop)
    N_active = args.n_rollout_threads - N_parent
    eval_interval = args.eval_interval
    save_node_interval = args.save_node_interval
    save_node = args.save_node
    historical_length = args.historical_length
    random.seed(args.seed)
    np.random.seed(args.seed)

    # region entity progression
    if args.scenario_name == 'simple_spread':
        phase_para = {'num_agents_current':8, 'num_mini_batch': 16}
    elif args.scenario_name == 'push_ball':
        phase_para = {'num_agents_current':4, 'num_mini_batch': 8}
    threshold_next = args.threshold_next
    decay_interval = args.decay_interval
    num_target = args.num_target
    num_agents_current = num_agents
    mix = False
    decay_episode = 0 # from 0 to decay_interval
    ratio_current = 1.0
    ratio_last = (1.0 - ratio_current) * 2
    node_last = None
    node_current = node_buffer(args=args, phase_num_agents=num_agents_current)
    # end region
    
    # run
    begin = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    current_timestep = 0
    active_length = args.n_rollout_threads
    starts_length = args.n_rollout_threads
    train_infos = {}

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           

        # entity progression
        if decay_episode == decay_interval and mix:
            decay_episode = 0
            ratio_current += 0.1
            ratio_current = min(ratio_current,1.0)
            ratio_last = (1.0 - ratio_current) * 2
            # turn off mix
            if np.abs(ratio_current - 1.0) <= 1e-5:
                mix = False

        # reproduction
        if mix:
            node_last.reproduction_num = round(args.B_exp * ratio_last)
            node_current.reproduction_num = round(args.B_exp * ratio_current)
        else:
            if node_last is not None:
                node_last.reproduction_num = args.B_exp
            node_current.reproduction_num = args.B_exp
        if node_last is not None and mix:
            node_last.archive += node_last.Sample_gradient(node_last.parent, current_timestep, use_gradient_noise=True)
        node_current.archive += node_current.Sample_gradient(node_current.parent, current_timestep, use_gradient_noise=True)

        # reset env 
        if mix:
            starts_last, active_length_last, starts_length_last = node_last.sample_tasks(round(N_active * ratio_last),round(N_parent * ratio_last))
            starts_current, active_length_current, starts_length_current = node_current.sample_tasks(round(N_active * ratio_current),round(N_parent * ratio_current))
            node_last.eval_score = np.zeros(shape=active_length_last)
        else:
            starts_current, active_length_current, starts_length_current = node_current.sample_tasks(N_active,N_parent)
        node_current.eval_score = np.zeros(shape=active_length_current)

        if mix:
            decay_episode += 1
            rollouts_last, current_timestep = collect_data(envs, agents, agents.actor_critic, args, node_last, starts_last, starts_length_last, active_length_last, current_timestep, train_infos)
        rollouts_current, current_timestep = collect_data(envs, agents, agents.actor_critic, args, node_current, starts_current, starts_length_current, active_length_current, current_timestep, train_infos)

        # update the network
        if args.share_policy:
            actor_critic.train()
            if mix:
                train_infos['Type of agents'] = 2
                value_loss, action_loss, dist_entropy = agents.update_double_share(node_last.num_agents, node_current.num_agents, rollouts_last, rollouts_current, current_timestep)
                rew = []
                for i in range(rollouts_last.rewards.shape[1]):
                    rew.append(np.sum(rollouts_last.rewards[:,i]))
                train_infos['train_episode_reward_last'] = np.mean(rew)
                # clean the buffer and reset
                rollouts_last.after_update()
            else:
                train_infos['Type of agents'] = 1
                value_loss, action_loss, dist_entropy = agents.update_share(node_current.num_agents, rollouts_current, current_timestep, warm_up=False)
            rew = []
            for i in range(rollouts_current.rewards.shape[1]):
                rew.append(np.sum(rollouts_current.rewards[:,i]))
            train_infos['train_episode_reward_current'] = np.mean(rew)
            train_infos['value_loss'] = value_loss
            # clean the buffer and reset
            rollouts_current.after_update()
        else:
            # TODO
            value_losses = []
            action_losses = []
            dist_entropies = [] 
            
            for agent_id in range(num_agents):
                actor_critic[agent_id].train()
                value_loss, action_loss, dist_entropy = agents[agent_id].update_single(agent_id, rollouts[agent_id])
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                    
                rew = []
                for i in range(rollouts[agent_id].rewards.shape[1]):
                    rew.append(np.sum(rollouts[agent_id].rewards[:,i]))
                train_infos['average_episode_reward'] = np.mean(rew)
                        
                rollouts[agent_id].after_update()

        # update nodes
        if mix:
            node_last.eval_score = node_last.eval_score
            archive_length, parent_length, del_easy_num, del_hard_num = node_last.update_buffer(active_length_last, current_timestep)
            train_infos['archive_length_last'] = archive_length
            train_infos['parent_length_last'] = parent_length
            train_infos['del_easy_num_last'] = del_easy_num
            train_infos['del_hard_num_last'] = del_hard_num
            if (episode+1) % save_node_interval ==0 and save_node:
                node_last.save_node(save_node_dir, episode)

        node_current.eval_score = node_current.eval_score
        archive_length, parent_length, del_easy_num, del_hard_num = node_current.update_buffer(active_length_current, current_timestep)
        train_infos['archive_length_current'] = archive_length
        train_infos['parent_length_current'] = parent_length
        train_infos['del_easy_num_current'] = del_easy_num
        train_infos['del_hard_num_current'] = del_hard_num
        if (episode+1) % save_node_interval ==0 and save_node:
            node_current.save_node(save_node_dir, episode)

        # region evaluation
        if episode % eval_interval==0:
            print('----------evaluation-------------')
            eval_num_agents = node_current.num_agents
            mean_cover_rate_current, eval_episode_reward_current = evaluation(envs, agents.actor_critic, args, eval_num_agents, current_timestep)
            train_infos['eval_episode_reward'] = eval_episode_reward_current
            train_infos['eval_cover_rate'] = mean_cover_rate_current
            print('eval_cover_rate', mean_cover_rate_current)

        # region entity curriculum
        if mean_cover_rate_current >= threshold_next and node_current.num_agents < num_target:
            mean_cover_rate_current = 0
            num_last = node_current.num_agents
            num_current = min(num_last * 2,num_target)
            add_num = num_current - num_last
            if add_num!=0:
                if args.share_policy:
                    mix = True
                    # save model when switch
                    save(str(save_dir) + "/%iagent_model_"%node_current.num_agents + str(threshold_next) + '.pt' , agents)
                    # set node_last and node_current
                    node_last = copy.deepcopy(node_current)
                    node_current = node_buffer(args=args, phase_num_agents=num_agents_current)
                    ratio_current = 0.1
                    ratio_last = (1.0 - ratio_current) * 2
                    # ppo
                    agents.actor_critic.num_agents = node_current.num_agents
                    if node_current.num_agents == phase_para['num_agents_current']:
                        agents.num_mini_batch = phase_para['num_mini_batch']
                else:
                    pass
                    # TODO
        # end region

        # region turn off mix 
        if mix:
            if mean_cover_rate_current > threshold_next and node_current.num_agents >= num_target:
                print('Turn off entity progression')
                mix = False
        # end region

        total_num_steps = current_timestep

        # log information
        if episode % args.log_interval == 0:
            end = time.time()
            log_infos(args, train_infos, current_timestep ,logger)
            print("\n Scenario {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(args.scenario_name,
                        args.algorithm_name,
                        episode, 
                        episodes,
                        total_num_steps,
                        args.num_env_steps,
                        int(total_num_steps / (end - begin))))
            if args.share_policy:
                print("value loss of agent: " + str(value_loss))
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id]))
                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
if __name__ == "__main__":
    main()
