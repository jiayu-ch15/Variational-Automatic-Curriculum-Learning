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
from algorithm.ppo import PPO
from algorithm.model import Policy, ATTBase_add

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage, RolloutStorage_share
from utils.single_storage import SingleRolloutStorage
import shutil
import numpy as np
import itertools
import pdb

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "MPE":
                env = MPEEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            # np.random.seed(args.seed + rank * 1000)
            return env
        return init_env
    if args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])

def main():
    args = get_config()
    
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
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    logger = SummaryWriter(str(log_dir)) 

    # env
    envs = make_parallel_env(args)
    num_agents = args.num_agents
    #Policy network
    if args.share_policy:
        share_base = ATTBase_add(envs.observation_space[0].shape[0], num_agents)
        actor_critic = Policy(envs.observation_space[0], 
                    envs.action_space[0],
                    num_agents = num_agents,
                    base=share_base,
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
                   
        #replay buffer
        rollouts = RolloutStorage_share(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0], 
                    envs.action_space[0],
                    args.hidden_size)        
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
    
    # reset env 
    obs, _ = envs.reset(num_agents)
    
    # replay buffer 
    if args.share_policy: 
        share_obs = obs.reshape(args.n_rollout_threads, -1)        
        # share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
        rollouts.share_obs[0] = share_obs.copy() 
        rollouts.obs[0] = obs.copy()               
        rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
        rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    else:
        
        share_obs = []
        for o in obs:
            share_obs.append(list(itertools.chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(num_agents):    
            rollouts[agent_id].share_obs[0] = share_obs.copy()
            rollouts[agent_id].obs[0] = np.array(list(obs[:,agent_id])).copy()               
            rollouts[agent_id].recurrent_hidden_states = np.zeros(rollouts[agent_id].recurrent_hidden_states.shape).astype(np.float32)
            rollouts[agent_id].recurrent_hidden_states_critic = np.zeros(rollouts[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)
    
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    timesteps = 0

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           

        step_cover_rate = np.zeros(shape=(args.n_rollout_threads, args.episode_length))
        for step in range(args.episode_length):
            # Sample actions
            values = []
            actions= []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            
            with torch.no_grad():                
                for agent_id in range(num_agents):
                    if args.share_policy:
                        actor_critic.eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                            # torch.FloatTensor(rollouts.share_obs[step,:,agent_id]),
                            torch.FloatTensor(rollouts.share_obs[step]), 
                            torch.FloatTensor(rollouts.obs[step,:,agent_id]), 
                            torch.FloatTensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.FloatTensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.FloatTensor(rollouts.masks[step,:,agent_id]))
                    else:
                        actor_critic[agent_id].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                            torch.FloatTensor(rollouts[agent_id].share_obs[step,:]), 
                            torch.FloatTensor(rollouts[agent_id].obs[step,:]), 
                            torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states[step,:]), 
                            torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states_critic[step,:]),
                            torch.FloatTensor(rollouts[agent_id].masks[step,:]))
                        
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    action_log_probs.append(action_log_prob.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
            
            # rearrange action
            actions_env = []
            for i in range(args.n_rollout_threads):
                one_hot_action_env = []
                for agent_id in range(num_agents):
                    if envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        uc_action = []
                        for j in range(envs.action_space[agent_id].shape):
                            uc_one_hot_action = np.zeros(envs.action_space[agent_id].high[j]+1)
                            uc_one_hot_action[actions[agent_id][i][j]] = 1
                            uc_action.append(uc_one_hot_action)
                        uc_action = np.concatenate(uc_action)
                        one_hot_action_env.append(uc_action)
                            
                    elif envs.action_space[agent_id].__class__.__name__ == 'Discrete':    
                        one_hot_action = np.zeros(envs.action_space[agent_id].n)
                        one_hot_action[actions[agent_id][i]] = 1
                        one_hot_action_env.append(one_hot_action)
                    else:
                        raise NotImplementedError
                actions_env.append(one_hot_action_env)
               
            # Obser reward and next obs
            obs, rewards, dones, infos, _ = envs.step(actions_env, args.n_rollout_threads, num_agents)
            step_cover_rate[:,step] = np.array(infos)[:,0]

            # If done then clean the history of observations.
            # insert data in buffer
            masks = []
            for i, done in enumerate(dones): 
                mask = []               
                for agent_id in range(num_agents): 
                    if done[agent_id]:    
                        recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                        recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                        mask.append([0.0])
                    else:
                        mask.append([1.0])
                masks.append(mask)
                            
            if args.share_policy: 
                share_obs = obs.reshape(args.n_rollout_threads, -1)        
                # share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
                
                rollouts.insert(share_obs, 
                            obs, 
                            np.array(recurrent_hidden_statess).transpose(1,0,2), 
                            np.array(recurrent_hidden_statess_critic).transpose(1,0,2), 
                            np.array(actions).transpose(1,0,2),
                            np.array(action_log_probs).transpose(1,0,2), 
                            np.array(values).transpose(1,0,2),
                            rewards, 
                            masks)
            else:
                share_obs = []
                for o in obs:
                    share_obs.append(list(itertools.chain(*o)))
                share_obs = np.array(share_obs)
                for agent_id in range(num_agents):
                    rollouts[agent_id].insert(share_obs, 
                            np.array(list(obs[:,agent_id])), 
                            np.array(recurrent_hidden_statess[agent_id]), 
                            np.array(recurrent_hidden_statess_critic[agent_id]), 
                            np.array(actions[agent_id]),
                            np.array(action_log_probs[agent_id]), 
                            np.array(values[agent_id]),
                            rewards[:,agent_id], 
                            np.array(masks)[:,agent_id])
        # import pdb;pdb.set_trace()
        logger.add_scalars('agent/cover_rate_1step',{'cover_rate_1step': np.mean(step_cover_rate[:,-1])},(episode + 1) * args.episode_length * args.n_rollout_threads)
        logger.add_scalars('agent/cover_rate_5step',{'cover_rate_5step': np.mean(np.mean(step_cover_rate[:,-5:],axis=1))},(episode + 1) * args.episode_length * args.n_rollout_threads)
                                      
        with torch.no_grad(): 
            for agent_id in range(num_agents):         
                if args.share_policy: 
                    actor_critic.eval()                
                    next_value,_,_ = actor_critic.get_value(agent_id,
                                                #    torch.FloatTensor(rollouts.share_obs[-1,:,agent_id]),
                                                   torch.FloatTensor(rollouts.share_obs[-1]), 
                                                   torch.FloatTensor(rollouts.obs[-1,:,agent_id]), 
                                                   torch.FloatTensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                   torch.FloatTensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                   torch.FloatTensor(rollouts.masks[-1,:,agent_id]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts.compute_returns(agent_id,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents.value_normalizer)
                else:
                    actor_critic[agent_id].eval()
                    next_value,_,_ = actor_critic[agent_id].get_value(agent_id,
                                                             torch.FloatTensor(rollouts[agent_id].share_obs[-1,:]), 
                                                             torch.FloatTensor(rollouts[agent_id].obs[-1,:]), 
                                                             torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states[-1,:]),
                                                             torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states_critic[-1,:]),
                                                             torch.FloatTensor(rollouts[agent_id].masks[-1,:]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts[agent_id].compute_returns(next_value, 
                                            args.use_gae, 
                                            args.gamma,
                                            args.gae_lambda, 
                                            args.use_proper_time_limits,
                                            args.use_popart,
                                            agents[agent_id].value_normalizer)

         
        # update the network
        if args.share_policy:
            actor_critic.train()
            value_loss, action_loss, dist_entropy = agents.update_share(num_agents, rollouts)
                           
            rew = []
            for i in range(rollouts.rewards.shape[1]):
                rew.append(np.sum(rollouts.rewards[:,i]))
            logger.add_scalars('average_episode_reward',
                {'average_episode_reward': np.mean(rew)},
                (episode + 1) * args.episode_length * args.n_rollout_threads)
            # clean the buffer and reset
            rollouts.after_update()
        else:
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
                logger.add_scalars('agent%i/average_episode_reward'%agent_id,
                    {'average_episode_reward': np.mean(rew)},
                    (episode + 1) * args.episode_length * args.n_rollout_threads)
                
                rollouts[agent_id].after_update()

        obs, _ = envs.reset(num_agents)
        # replay buffer 
        if args.share_policy: 
            share_obs = obs.reshape(args.n_rollout_threads, -1)        
            # share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
            rollouts.share_obs[0] = share_obs.copy() 
            rollouts.obs[0] = obs.copy()               
            rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
            rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
        else:
            
            share_obs = []
            for o in obs:
                share_obs.append(list(itertools.chain(*o)))
            share_obs = np.array(share_obs)
            for agent_id in range(num_agents):    
                rollouts[agent_id].share_obs[0] = share_obs.copy()
                rollouts[agent_id].obs[0] = np.array(list(obs[:,agent_id])).copy()               
                rollouts[agent_id].recurrent_hidden_states = np.zeros(rollouts[agent_id].recurrent_hidden_states.shape).astype(np.float32)
                rollouts[agent_id].recurrent_hidden_states_critic = np.zeros(rollouts[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)
                                                                     
        total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads

        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(save_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'model': actor_critic[agent_id]
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
                        int(total_num_steps / (end - start))))
            if args.share_policy:
                print("value loss of agent: " + str(value_loss))
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id]))

            # if args.env_name == "MPE":
            #     for agent_id in range(num_agents):
            #         show_rewards = []
            #         for info in infos:                        
            #             if 'individual_reward' in info[agent_id].keys():
            #                 show_rewards.append(info[agent_id]['individual_reward'])                    
            #         logger.add_scalars('agent%i/individual_reward' % agent_id, {'individual_reward': np.mean(show_rewards)}, total_num_steps)
                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
if __name__ == "__main__":
    main()
