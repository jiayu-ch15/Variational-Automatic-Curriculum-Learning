#!/usr/bin/env python
import sys
sys.path.append("..")
import copy
import glob
import os
import time
import numpy as np
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs.hns import BlueprintConstructionEnv, BoxLockingEnv, ShelterConstructionEnv, HideAndSeekEnv
from algorithm.autocurriculum import node_buffer, log_infos
from algorithm.ppo import PPO
from algorithm.hns_model import Policy

from config import get_config
from utils.env_wrappers import SimplifySubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
import shutil
import numpy as np
from utils.multi_discrete import MultiDiscrete
from functools import reduce
from scipy.spatial.distance import cdist
import pdb
import wandb

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "BlueprintConstruction":
                env = BlueprintConstructionEnv(args)
            elif args.env_name == "BoxLocking":
                env = BoxLockingEnv(args)
            elif args.env_name == "ShelterConstruction":
                env = ShelterConstructionEnv(args)
            elif args.env_name == "hidenseek":
                env = HideAndSeekEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    return SimplifySubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def make_eval_env(args, num_thread):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "BlueprintConstruction":
                env = BlueprintConstructionEnv(args)
            elif args.env_name == "BoxLocking":
                env = BoxLockingEnv(args)
            elif args.env_name == "ShelterConstruction":
                env = ShelterConstructionEnv(args)
            elif args.env_name == "hidenseek":
                env = HideAndSeekEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    return SimplifySubprocVecEnv([get_env_fn(i) for i in range(num_thread)])

def handle_dict_obs(keys, order_obs, mask_order_obs, dict_obs, num_agents, num_hiders):
    obs = []
    share_obs = []  
    for d_o in dict_obs:
        for i, key in enumerate(order_obs):
            if key in keys:             
                if mask_order_obs[i] == None:
                    temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = d_o[key].reshape(num_agents,-1).copy()
                    temp_mask = d_o[mask_order_obs[i]].copy()
                    temp_obs = d_o[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        obs.append(reshape_obs)
        share_obs.append(reshape_share_obs)   
    obs = np.array(obs)[:,num_hiders:]
    share_obs = np.array(share_obs)[:,num_hiders:]
    return obs, share_obs

def main():
    args = get_config()
    if args.use_wandb:
        run = wandb.init(project=args.scenario_name,name=str(args.algorithm_name) + "_seed" + str(args.seed))

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
    envs = make_parallel_env(args)
    if args.eval:
        eval_num = 100
        eval_env = make_eval_env(args,eval_num)
    
    num_hiders = args.num_hiders
    num_seekers = args.num_seekers
    num_agents = num_hiders + num_seekers
    num_boxes = args.num_boxes
    num_ramps = args.num_ramps
    all_action_space = []
    all_obs_space = []
    action_movement_dim = []

    # handle dict_obs
    order_obs = ['agent_qpos_qvel', 'box_obs','ramp_obs','construction_site_obs', 'observation_self']    
    mask_order_obs = [None, None, None, None, None]
    keys = envs.observation_space.spaces.keys()
    for agent_id in range(num_agents):
        # deal with dict action space
        action_movement = envs.action_space['action_movement'][agent_id].nvec
        action_movement_dim.append(len(action_movement))      
        action_glueall = envs.action_space['action_glueall'][agent_id].n
        action_vec = np.append(action_movement, action_glueall)
        if 'action_pull' in envs.action_space.spaces.keys():
            action_pull = envs.action_space['action_pull'][agent_id].n
            action_vec = np.append(action_vec, action_pull)
        action_space = MultiDiscrete([[0,vec-1] for vec in action_vec])
        all_action_space.append(action_space) 
        # deal with dict obs space
        obs_space = []
        obs_dim = 0
        for key in order_obs:
            if key in envs.observation_space.spaces.keys():
                space = list(envs.observation_space[key].shape)
                if len(space)<2:  
                    space.insert(0,1)        
                obs_space.append(space)
                obs_dim += reduce(lambda x,y:x*y,space)
        obs_space.insert(0,obs_dim)
        all_obs_space.append(obs_space)

    if args.share_policy:
        actor_critic = Policy(all_obs_space[0], 
                    all_action_space[0],
                    num_agents = num_seekers,
                    with_PC = False,
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
                                 'use_ReLU':args.use_ReLU,
                                 'use_same_dim':True
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
        for agent_id in range(num_seekers):
            ac = Policy(all_obs_space[0], 
                      all_action_space[0],
                      num_agents = num_seekers,
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
                                 'use_ReLU':args.use_ReLU,
                                 'use_same_dim':True
                                 },
                      device = device)
            ac.to(device)
            # algorithm
            agent = PPO_merge(ac,
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
        rollouts = RolloutStorage(num_seekers,
                    args.episode_length, 
                    args.n_rollout_threads,
                    all_obs_space[0], 
                    all_action_space[0],
                    args.hidden_size,
                    use_same_dim=True)

    starts = []
    N_parent = int(args.n_rollout_threads * args.sol_prop)
    N_active = args.n_rollout_threads - N_parent
    eval_interval = args.eval_interval
    save_node_interval = args.save_node_interval
    save_node = args.save_node
    historical_length = args.historical_length
    eval_number = args.eval_number
    random.seed(args.seed)
    np.random.seed(args.seed)
    node = node_buffer( args=args, phase_num_agents=num_agents)

    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    timesteps = 0
    current_timestep = 0
    active_length = args.n_rollout_threads
    starts_length = args.n_rollout_threads
    train_infos = {}

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_seekers):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           
        
        # reproduction
        node.archive += node.Sample_gradient_hns(node.parent, current_timestep, use_gradient_noise=True)
        
        # info list
        discard_episode = 0

        # reset env 
        starts, active_length, starts_length = node.sample_tasks(N_active,N_parent)
        node.eval_score = np.zeros(shape=active_length)

        for times in range(eval_number):
            dict_obs = envs.init_hidenseek(starts,starts_length)
            obs, share_obs = handle_dict_obs(keys, order_obs, mask_order_obs, dict_obs, num_agents, num_hiders)
            rollouts = RolloutStorage(num_seekers,
                        args.episode_length, 
                        starts_length,
                        all_obs_space[0], 
                        all_action_space[0],
                        args.hidden_size,
                        use_same_dim=True)         
            
            # replay buffer  
            rollouts.share_obs[0] = share_obs.copy() 
            rollouts.obs[0] = obs.copy()                
            rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
            rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
            step_cover_rate = np.zeros(shape=(active_length,args.episode_length))
            for step in range(args.episode_length):
                # Sample actions
                values = []
                actions= []
                action_log_probs = []
                recurrent_hidden_statess = []
                recurrent_hidden_statess_critic = []
                with torch.no_grad():                
                    for agent_id in range(num_seekers):
                        if args.share_policy:
                            actor_critic.eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id+num_hiders,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]))
                        else:
                            actor_critic[agent_id].eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id+num_hiders,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]))
                            
                        values.append(value.detach().cpu().numpy())
                        actions.append(action.detach().cpu().numpy())
                        action_log_probs.append(action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
                # rearrange action        
                actions_env = []
                for n_rollout_thread in range(starts_length):
                    action_movement = []
                    action_pull = []
                    action_glueall = []
                    for k in range(num_hiders):
                        #action_movement.append(np.random.randint(11, size=3))  #hider随机游走
                        action_movement.append(np.array([5,5,5]))   #hider静止不动
                        action_pull.append(0)
                        action_glueall.append(0)
                    for k in range(num_seekers):
                        action_movement.append(actions[k][n_rollout_thread][:3])
                        action_pull.append(np.int(actions[k][n_rollout_thread][3]))
                        action_glueall.append(np.int(actions[k][n_rollout_thread][4]))
                    action_movement = np.stack(action_movement, axis = 0)
                    action_glueall = np.stack(action_glueall, axis = 0)
                    action_pull = np.stack(action_pull, axis = 0)                        
                    one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
                    actions_env.append(one_env_action)
                        
                # Obser reward and next obs
                dict_obs, rewards, dones, infos = envs.step(actions_env)
                for env_id in range(step_cover_rate.shape[0]):
                    step_cover_rate[env_id, step] = infos[env_id]['success_rate']
                rewards=rewards[:, num_hiders:, np.newaxis]            

                # If done then clean the history of observations.
                # insert data in buffer
                masks = []
                for i, done in enumerate(dones): 
                    if done:
                        if "discard_episode" in infos[i].keys():
                            if infos[i]['discard_episode']:
                                discard_episode += 1
                    mask = []               
                    for agent_id in range(num_seekers): 
                        if done:    
                            recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                            recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                            mask.append([0.0])                        
                        else:
                            mask.append([1.0])
                    masks.append(mask)                            

                obs, share_obs = handle_dict_obs(keys, order_obs, mask_order_obs, dict_obs, num_agents, num_hiders)
        
                rollouts.insert(share_obs, 
                                obs, 
                                np.array(recurrent_hidden_statess).transpose(1,0,2), 
                                np.array(recurrent_hidden_statess_critic).transpose(1,0,2), 
                                np.array(actions).transpose(1,0,2),
                                np.array(action_log_probs).transpose(1,0,2), 
                                np.array(values).transpose(1,0,2),
                                rewards, 
                                masks)
            train_infos['training_success_rate'] = np.mean(step_cover_rate[:, -historical_length:])
            current_timestep += args.episode_length * starts_length
            node.eval_score += np.mean(step_cover_rate[:,-historical_length:],axis=1)
            with torch.no_grad(): 
                for agent_id in range(num_seekers):         
                    if args.share_policy: 
                        actor_critic.eval()                
                        next_value,_,_ = actor_critic.get_value(agent_id,
                                                    torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                    torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                    torch.tensor(rollouts.masks[-1,:,agent_id]))
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
                                                    torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.obs[-1,:,agent_id]), 
                                                    torch.tensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                    torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                    torch.tensor(rollouts.masks[-1,:,agent_id]))
                        next_value = next_value.detach().cpu().numpy()
                        rollouts.compute_returns(agent_id,
                                        next_value, 
                                        args.use_gae, 
                                        args.gamma,
                                        args.gae_lambda, 
                                        args.use_proper_time_limits,
                                        args.use_popart,
                                        agents[agent_id].value_normalizer)
            
            # update the network
            if args.share_policy:
                actor_critic.train()
                value_loss, action_loss, dist_entropy = agents.update_share(num_seekers, rollouts, warm_up = False)
                train_infos['value_loss'] = value_loss
                train_infos['train_reward'] = np.mean(rollouts.rewards)
            else:
                value_losses = []
                action_losses = []
                dist_entropies = [] 
                
                for agent_id in range(num_seekers):
                    actor_critic[agent_id].train()
                    value_loss, action_loss, dist_entropy = agents[agent_id].update(agent_id, rollouts)
                    value_losses.append(value_loss)
                    action_losses.append(action_loss)
                    dist_entropies.append(dist_entropy)

                    train_infos['value_loss'] = value_loss
                    train_infos['agent%i/reward' % agent_id] = np.mean(rollouts.rewards[:,:,agent_id])
                                                                   
            # clean the buffer and reset
            rollouts.after_update()

        # move nodes
        node.eval_score = node.eval_score / eval_number
        archive_length, parent_length, del_easy_num, del_hard_num = node.update_buffer(active_length, current_timestep)
        train_infos['archive_length'] = archive_length
        train_infos['parent_length'] = parent_length
        train_infos['del_easy_num'] = del_easy_num
        train_infos['del_hard_num'] = del_hard_num
        if (episode+1) % save_node_interval ==0 and save_node:
            node.save_node(save_node_dir, episode)

        # eval 
        if episode % args.eval_interval == 0 and args.eval:
            dict_obs = eval_env.reset()
            episode_length = args.env_horizon
            obs, share_obs = handle_dict_obs(keys, order_obs, mask_order_obs, dict_obs, num_agents, num_hiders)
            rollouts = RolloutStorage(num_seekers,
                        episode_length, 
                        eval_num,
                        all_obs_space[0], 
                        all_action_space[0],
                        args.hidden_size,
                        use_same_dim=True)
            rollouts.share_obs[0] = share_obs.copy() 
            rollouts.obs[0] = obs.copy()                
            rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
            rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
            for step in range(episode_length):
                # Sample actions
                values = []
                actions= []
                action_log_probs = []
                recurrent_hidden_statess = []
                recurrent_hidden_statess_critic = []
                with torch.no_grad():                
                    for agent_id in range(num_seekers):
                        if args.share_policy:
                            actor_critic.eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id+num_hiders,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]),
                            None,
                            deterministic=True)
                        else:
                            actor_critic[agent_id+num_seekers].eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                            torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.tensor(rollouts.obs[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.tensor(rollouts.masks[step,:,agent_id]))
                            
                        values.append(value.detach().cpu().numpy())
                        actions.append(action.detach().cpu().numpy())
                        action_log_probs.append(action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
                
                # rearrange action          
                actions_env = []
                for n_rollout_thread in range(eval_num):
                    action_movement = []
                    action_pull = []
                    action_glueall = []
                    for k in range(num_hiders):
                        #action_movement.append(np.random.randint(11, size=3))  #hider随机游走
                        action_movement.append(np.array([5,5,5]))   #hider静止不动
                        action_pull.append(0)
                        action_glueall.append(0)
                    for k in range(num_seekers):
                        action_movement.append(actions[k][n_rollout_thread][:3])
                        action_pull.append(np.int(actions[k][n_rollout_thread][3]))
                        action_glueall.append(np.int(actions[k][n_rollout_thread][4]))
                    action_movement = np.stack(action_movement, axis = 0)
                    action_glueall = np.stack(action_glueall, axis = 0)
                    action_pull = np.stack(action_pull, axis = 0)                         
                    one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
                    actions_env.append(one_env_action)
                        
                # Obser reward and next obs
                dict_obs, rewards, dones, infos = eval_env.step(actions_env)
                rewards=rewards[:, num_hiders:, np.newaxis]          

                # If done then clean the history of observations.
                # insert data in buffer
                masks = []
                for i, done in enumerate(dones): 
                    if done:
                        if "discard_episode" in infos[i].keys():
                            if infos[i]['discard_episode']:
                                discard_episode += 1
                    mask = []               
                    for agent_id in range(num_seekers): 
                        if done:    
                            recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                            recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                            mask.append([0.0])                        
                        else:
                            mask.append([1.0])
                    masks.append(mask) 
                                               
                obs, share_obs = handle_dict_obs(keys, order_obs, mask_order_obs, dict_obs, num_agents, num_hiders)
        
                rollouts.insert(share_obs, 
                                obs, 
                                np.array(recurrent_hidden_statess).transpose(1,0,2), 
                                np.array(recurrent_hidden_statess_critic).transpose(1,0,2), 
                                np.array(actions).transpose(1,0,2),
                                np.array(action_log_probs).transpose(1,0,2), 
                                np.array(values).transpose(1,0,2),
                                rewards, 
                                masks)  
            sum_cover_rate = 0
            for i in range(len(infos)):
                sum_cover_rate += infos[i]['success_rate'] 
            sum_cover_rate = sum_cover_rate / len(infos)
            print('eval_success_rate: ', sum_cover_rate)
            train_infos['eval_success_rate'] =  sum_cover_rate
            train_infos['eval_reward'] = np.mean(rollouts.rewards)

        total_num_steps = current_timestep
        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(save_dir) + "/agent_model_" + str(episode) + ".pt")
            else:
                for agent_id in range(num_seekers):                                                  
                    torch.save({
                                'model': actor_critic[agent_id]
                                }, 
                                str(save_dir) + "/agent%i_model" % agent_id + ".pt")

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
                        int(total_num_steps / (end - start))))
            if args.share_policy:
                print("value loss of agent: " + str(value_loss))
                print("reward of agent: ", np.mean(rollouts.rewards))
            else:
                for agent_id in range(num_seekers):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id])) 

            train_infos['discard_episode'] = discard_episode          
        
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
    if args.eval:
        eval_env.close()
if __name__ == "__main__":
    main()
