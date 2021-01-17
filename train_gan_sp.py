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
from algorithm.model import Policy,Policy3, ATTBase_add, ATTBase_actor_dist_add, ATTBase_critic_add

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
from utils.single_storage import SingleRolloutStorage

from gan.lsgan import LSGAN
from gan.utils import StateCollection

import shutil
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import random
import copy
import matplotlib.pyplot as plt
import pdb
import wandb
# wandb.init(project="my-project")
np.set_printoptions(linewidth=1000)

gan_configs = {
    'cuda':True,
    'batch_size': 64,
    'gan_outer_iters':300,
    'num_labels' : 1,       # normally only 1 label(success or not)  
    'goal_size': 0,
    'goal_center': [0,0],
    'goal_range': 0,
    'goal_noise_level': [0.03],
    'gan_noise_size': 16,
    'lr': 0.001,
}

goal_configs ={
    'num_new_goals': 350,
    'num_old_goals': 150,
    'coll_eps': 0.3,
    'R_min': 0.5,
    'R_max': 0.95,
    'historical_length': 5,
}

import warnings
warnings.filterwarnings("ignore")

class StateGenerator(object):
    """A base class for state generation."""

    def pretrain_uniform(self):
        """Pretrain the generator distribution to uniform distribution in the limit."""
        raise NotImplementedError

    def pretrain(self, states):
        """Pretrain with state distribution in the states list."""
        raise NotImplementedError

    def sample_states(self, size):
        """Sample states with given size."""
        raise NotImplementedError

    def sample_states_with_noise(self, size):
        """Sample states with noise."""
        raise NotImplementedError

    def train(self, states, labels):
        """Train with respect to given states and labels."""
        raise NotImplementedError

class StateGAN(StateGenerator):
    """A GAN for generating states. """
    def __init__(self, gan_configs, state_range = None, state_bounds = None):
        self.gan = LSGAN(gan_configs)
        self.gan_configs = gan_configs
        self.state_size = gan_configs['goal_size']
        self.evaluater_size = gan_configs['num_labels']
        self.state_center = np.array(gan_configs['goal_center'])
        if state_range is not None:
            self.state_range = state_range
            self.state_bounds = np.vstack([-self.state_range * np.ones(self.state_size), self.state_range * np.ones(self.state_size)])
        elif state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
            self.state_range = self.state_bounds[1] - self.state_bounds[0]

        self.state_noise_level = gan_configs['goal_noise_level']
        # print('state_center is : ', self.state_center[0], 'state_range: ', self.state_range,
        #       'state_bounds: ', self.state_bounds)

    def pretrain_uniform(self, size=10000, report=None):
        """
        :param size: number of uniformly sampled states (that we will try to fit as output of the GAN)
        :param outer_iters: of the GAN
        """
        states = np.random.uniform(
            self.state_center + self.state_bounds[0], self.state_center + self.state_bounds[1], size=(size, self.state_size)
        )
        return self.pretrain(states)

    def pretrain(self, states, outer_iters=500, generator_iters=None, discriminator_iters=None):
        """
        Pretrain the state GAN to match the distribution of given states.
        :param states: the state distribution to match
        :param outer_iters: of the GAN
        """
        labels = np.ones((states.shape[0], self.evaluater_size))  # all state same label --> uniform
        return self.train(states, labels, outer_iters)

    def _add_noise_to_states(self, states):
        noise = np.random.randn(*states.shape) * self.state_noise_level
        states += noise
        return np.clip(states, self.state_center + self.state_bounds[0], self.state_center + self.state_bounds[1])

    def sample_states(self, size):  # un-normalizes the states
        normalized_states, noise = self.gan.sample_generator(size)
        tmp = normalized_states.cpu().detach().numpy()
        states = self.state_center + tmp * self.state_bounds[1]
        return states, noise

    def sample_states_with_noise(self, size):
        states, noise = self.sample_states(size)
        states = self._add_noise_to_states(states)
        return states, noise

    def train(self, states, labels, outer_iters=None, generator_iters=None, discriminator_iters=None):
        # 归一化到0~1
        normalized_states = (states - self.state_center) / self.state_bounds[1][0]
        return self.gan.train(normalized_states, labels, self.gan_configs)

    def discriminator_predict(self, states):
        return self.gan.discriminator_predict(states)

def generate_initial_goals(num_case, start_boundary, agent_num):
    pos_dim = 2  #坐标维度
    one_starts_agent = np.zeros((agent_num, pos_dim), dtype = float)
    one_starts_landmark = np.zeros((agent_num, pos_dim), dtype = float)
    goals = np.zeros((num_case, (agent_num + agent_num)*pos_dim), dtype = float)
    for j in range(num_case):
        for i in range(agent_num):
            landmark_location = np.array([np.random.uniform(start_boundary[0],start_boundary[1]),np.random.uniform(start_boundary[2],start_boundary[3])])
            one_starts_landmark[i] = landmark_location
        indices = random.sample(range(agent_num), agent_num)
        for  k in indices:
            epsilon = -2 * 0.01 * random.random() + 0.01
            one_starts_agent[k] = one_starts_landmark[k]+epsilon
        goals[j] = np.concatenate((one_starts_agent, one_starts_landmark), axis=None)
    return goals

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

def numpy_to_list(array, list_length, shape):
    res = []
    for i in range(list_length):
        sub_arr = array[i].reshape(shape)
        res.append(sub_arr)
    return res

def main():
    args = get_config()
    run = wandb.init(project='goal_gan_sp',name=str(args.algorithm_name) + "_seed" + str(args.seed))
    # run = wandb.init(project='check',name='separate_reward')
    
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
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   device=device)
                   
        # #replay buffer
        # rollouts = RolloutStorage(num_agents,
        #             args.episode_length, 
        #             args.n_rollout_threads,
        #             envs.observation_space[0], 
        #             envs.action_space[0],
        #             args.hidden_size)        
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
    
   
    boundary = 3
    start_boundary = [-0.3,0.3,-0.3,0.3] # 分别代表x的范围和y的范围
    max_step = 0.6
    N_easy = 0
    test_flag = 0
    reproduce_flag = 0
    target_num = 4
    # last_agent_num = 4
    # now_agent_num = num_agents
    test_num_agents = 8
    mean_cover_rate = 0
    eval_frequency = 2 #需要fix几个回合
    check_frequency = 1
    save_node_frequency = 5
    save_node_flag = False
    save_90_flag = False
    historical_length = 5
    random.seed(args.seed)
    np.random.seed(args.seed)


    # init the Gan
    gan_configs['goal_range'] = boundary
    gan_configs['goal_center'] = np.zeros((num_agents + num_agents)* 2, dtype=float)
    gan_configs['goal_size'] = (num_agents + num_agents)*2
    gan = StateGAN(gan_configs = gan_configs, state_range=gan_configs['goal_range'])
    feasible_goals = generate_initial_goals(num_case = 10000, start_boundary = start_boundary, agent_num = args.num_agents)                            
    dis_loss, gen_loss = gan.pretrain(states=feasible_goals, outer_iters=gan_configs['gan_outer_iters'])
    print('discriminator_loss:',str(dis_loss.cpu()), 'generator_loss:',str(gen_loss.cpu()))
    
    # init the StateCollection
    all_goals = StateCollection(distance_threshold=goal_configs['coll_eps'])

    # run
    begin = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads // eval_frequency
    curriculum_episode = 0
    current_timestep = 0
    one_length = args.n_rollout_threads
    starts_length = args.n_rollout_threads
    num_envs = args.n_rollout_threads

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           



        raw_goals, _ = gan.sample_states_with_noise(goal_configs['num_new_goals'])
        # replay buffer
        if all_goals.size > 0:
            old_goals = all_goals.sample(goal_configs['num_old_goals'])
            goals = np.vstack([raw_goals, old_goals])
        else:
            goals = raw_goals   
        if goals.shape[0] < num_envs:
            add_num = num_envs - goals.shape[0]
            goals = np.vstack([goals, goals[:add_num]]) #补齐到num_new_goals+num_old_goals   
        # generate the starts
        starts = numpy_to_list(goals, list_length=num_envs, shape=(num_agents*2,2))

        for times in range(eval_frequency):
            actor_critic.agents_num = num_agents
            obs = envs.new_starts_obs(starts, num_agents, starts_length)
            #replay buffer
            rollouts = RolloutStorage(num_agents,
                        args.episode_length, 
                        starts_length,
                        envs.observation_space[0], 
                        envs.action_space[0],
                        args.hidden_size) 
            # replay buffer init
            if args.share_policy: 
                share_obs = obs.reshape(starts_length, -1)        
                share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
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
            step_cover_rate = np.zeros(shape=(one_length,args.episode_length))
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
                                torch.FloatTensor(rollouts.share_obs[step,:,agent_id]), 
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
                for i in range(starts_length):
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
                obs, rewards, dones, infos, _ = envs.step(actions_env, starts_length, num_agents)
                cover_rate_list = []
                for env_id in range(one_length):
                    cover_rate_list.append(infos[env_id][0]['cover_rate'])
                step_cover_rate[:,step] = np.array(cover_rate_list)
                # step_cover_rate[:,step] = np.array(infos)[0:one_length,0]

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
                    share_obs = obs.reshape(starts_length, -1)        
                    share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
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
            # logger.add_scalars('agent/training_cover_rate',{'training_cover_rate': np.mean(np.mean(step_cover_rate[:,-historical_length:],axis=1))}, current_timestep)
            wandb.log({'training_cover_rate': np.mean(np.mean(step_cover_rate[:,-historical_length:],axis=1))}, current_timestep)
            print('training_cover_rate: ', np.mean(np.mean(step_cover_rate[:,-historical_length:],axis=1)))
            current_timestep += args.episode_length * starts_length
            curriculum_episode += 1
            
            #region train the gan

            if times == 1:
                start_time = time.time()
                filtered_raw_goals = []
                labels = np.zeros((num_envs, 1), dtype = int)
                for i in range(num_envs):
                    R_i = np.mean(step_cover_rate[i, -goal_configs['historical_length']:])
                    if R_i < goal_configs['R_max'] and R_i > goal_configs['R_min']:
                        labels[i] = 1
                        filtered_raw_goals.append(goals[i])
                gan.train(goals, labels)
                all_goals.append(filtered_raw_goals)
                end_time = time.time()
                print("Gan training time: %.2f"%(end_time-start_time))

            with torch.no_grad():  # get value and com
                for agent_id in range(num_agents):         
                    if args.share_policy: 
                        actor_critic.eval()                
                        next_value,_,_ = actor_critic.get_value(agent_id,
                                                    torch.FloatTensor(rollouts.share_obs[-1,:,agent_id]), 
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
                value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(num_agents, rollouts, False, initial_optimizer=False) 
                print('value_loss: ', value_loss)
                wandb.log(
                    {'value_loss': value_loss},
                    current_timestep)
                rew = []
                for i in range(rollouts.rewards.shape[1]):
                    rew.append(np.sum(rollouts.rewards[:,i]))
                wandb.log(
                    {'average_episode_reward': np.mean(rew)},
                    current_timestep)
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
                        (episode+1) * args.episode_length * one_length*eval_frequency)
                    
                    rollouts[agent_id].after_update()


        # test
        if episode % check_frequency==0:
            actor_critic.agents_num = test_num_agents
            obs, _ = envs.reset(test_num_agents)
            episode_length = 70
            #replay buffer
            rollouts = RolloutStorage(test_num_agents,
                        episode_length, 
                        args.n_rollout_threads,
                        envs.observation_space[0], 
                        envs.action_space[0],
                        args.hidden_size) 
            # replay buffer init
            if args.share_policy: 
                share_obs = obs.reshape(args.n_rollout_threads, -1)        
                share_obs = np.expand_dims(share_obs,1).repeat(test_num_agents,axis=1)    
                rollouts.share_obs[0] = share_obs.copy() 
                rollouts.obs[0] = obs.copy()               
                rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
                rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
            else:
                share_obs = []
                for o in obs:
                    share_obs.append(list(itertools.chain(*o)))
                share_obs = np.array(share_obs)
                for agent_id in range(test_num_agents):    
                    rollouts[agent_id].share_obs[0] = share_obs.copy()
                    rollouts[agent_id].obs[0] = np.array(list(obs[:,agent_id])).copy()               
                    rollouts[agent_id].recurrent_hidden_states = np.zeros(rollouts[agent_id].recurrent_hidden_states.shape).astype(np.float32)
                    rollouts[agent_id].recurrent_hidden_states_critic = np.zeros(rollouts[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)
            test_cover_rate = np.zeros(shape=(args.n_rollout_threads,episode_length))
            for step in range(episode_length):
                # Sample actions
                values = []
                actions= []
                action_log_probs = []
                recurrent_hidden_statess = []
                recurrent_hidden_statess_critic = []
                
                with torch.no_grad():                
                    for agent_id in range(test_num_agents):
                        if args.share_policy:
                            actor_critic.eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                                torch.FloatTensor(rollouts.share_obs[step,:,agent_id]), 
                                torch.FloatTensor(rollouts.obs[step,:,agent_id]), 
                                torch.FloatTensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                                torch.FloatTensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                                torch.FloatTensor(rollouts.masks[step,:,agent_id]),deterministic=True)
                        else:
                            actor_critic[agent_id].eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                                torch.FloatTensor(rollouts[agent_id].share_obs[step,:]), 
                                torch.FloatTensor(rollouts[agent_id].obs[step,:]), 
                                torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states[step,:]), 
                                torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states_critic[step,:]),
                                torch.FloatTensor(rollouts[agent_id].masks[step,:]),deterministic=True)
                            
                        values.append(value.detach().cpu().numpy())
                        actions.append(action.detach().cpu().numpy())
                        action_log_probs.append(action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
                
                # rearrange action
                actions_env = []
                for i in range(args.n_rollout_threads):
                    one_hot_action_env = []
                    for agent_id in range(test_num_agents):
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
                cover_rate_list = []
                for env_id in range(args.n_rollout_threads):
                    cover_rate_list.append(infos[env_id][0]['cover_rate'])
                test_cover_rate[:,step] = np.array(cover_rate_list)
                # test_cover_rate[:,step] = np.array(infos)[:,0]

                # If done then clean the history of observations.
                # insert data in buffer
                masks = []
                for i, done in enumerate(dones): 
                    mask = []               
                    for agent_id in range(test_num_agents): 
                        if done[agent_id]:    
                            recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                            recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                            mask.append([0.0])
                        else:
                            mask.append([1.0])
                    masks.append(mask)
                                
                if args.share_policy: 
                    share_obs = obs.reshape(args.n_rollout_threads, -1)        
                    share_obs = np.expand_dims(share_obs,1).repeat(test_num_agents,axis=1)    
                    
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
                    for agent_id in range(test_num_agents):
                        rollouts[agent_id].insert(share_obs, 
                                np.array(list(obs[:,agent_id])), 
                                np.array(recurrent_hidden_statess[agent_id]), 
                                np.array(recurrent_hidden_statess_critic[agent_id]), 
                                np.array(actions[agent_id]),
                                np.array(action_log_probs[agent_id]), 
                                np.array(values[agent_id]),
                                rewards[:,agent_id], 
                                np.array(masks)[:,agent_id])

            # logger.add_scalars('agent/cover_rate_1step',{'cover_rate_1step': np.mean(test_cover_rate[:,-1])},current_timestep)
            # logger.add_scalars('agent/cover_rate_5step',{'cover_rate_5step': np.mean(np.mean(test_cover_rate[:,-historical_length:],axis=1))}, current_timestep)
            rew = []
            for i in range(rollouts.rewards.shape[1]):
                rew.append(np.sum(rollouts.rewards[:,i]))
            wandb.log(
                {'eval_episode_reward': np.mean(rew)},
                current_timestep)
            wandb.log({str(test_num_agents) + 'cover_rate_1step': np.mean(test_cover_rate[:,-1])},current_timestep)
            wandb.log({str(test_num_agents) + 'cover_rate_5step': np.mean(np.mean(test_cover_rate[:,-historical_length:],axis=1))}, current_timestep)
            mean_cover_rate = np.mean(np.mean(test_cover_rate[:,-historical_length:],axis=1))
            if mean_cover_rate >= 0.9 and args.algorithm_name=='ours' and save_90_flag:
                torch.save({'model': actor_critic}, str(save_dir) + "/cover09_agent_model.pt")
                save_90_flag = False

        total_num_steps = current_timestep

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
