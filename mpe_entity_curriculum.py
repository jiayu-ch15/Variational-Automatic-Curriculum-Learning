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

class node_buffer():
    def __init__(self,num_agents,buffer_length,archive_initial_length,reproduction_num,max_step,start_boundary,boundary):
        self.num_agents = num_agents
        self.buffer_length = buffer_length
        self.archive = self.produce_good_case(archive_initial_length, start_boundary, self.num_agents)
        self.archive_novelty = self.get_novelty(self.archive,self.archive)
        self.archive, self.archive_novelty = self.novelty_sort(self.archive, self.archive_novelty)
        self.childlist = []
        self.hardlist = []
        self.parent = []
        self.parent_all = []
        self.max_step = max_step
        self.boundary = boundary
        self.reproduction_num = reproduction_num
        self.choose_child_index = []
        self.choose_archive_index = []
        self.eval_score = np.zeros(shape=len(self.archive))
        self.topk = 5

    def produce_good_case(self, num_case, start_boundary, now_agent_num):
        one_starts_landmark = []
        one_starts_agent = []
        archive = [] 
        for j in range(num_case):
            for i in range(now_agent_num):
                landmark_location = np.random.uniform(-start_boundary, +start_boundary, 2) 
                one_starts_landmark.append(copy.deepcopy(landmark_location))
            # index_sample = BatchSampler(SubsetRandomSampler(range(now_agent_num)),now_agent_num,drop_last=True)
            indices = random.sample(range(now_agent_num), now_agent_num)
            for k in indices:
                epsilon = -2 * 0.01 * random.random() + 0.01
                one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
            # select_starts.append(one_starts_agent+one_starts_landmark)
            archive.append(one_starts_agent+one_starts_landmark)
            one_starts_agent = []
            one_starts_landmark = []
        return archive

    def produce_good_case_grid(self, num_case, start_boundary, now_agent_num):
        # agent_size=0.1
        cell_size = 0.2
        grid_num = int(start_boundary * 2 / cell_size) + 1
        grid = np.zeros(shape=(grid_num,grid_num))
        one_starts_landmark = []
        one_starts_landmark_grid = []
        one_starts_agent = []
        archive = [] 
        for j in range(num_case):
            for i in range(now_agent_num):
                while 1:
                    landmark_location_grid = np.random.randint(0, grid.shape[0], 2) 
                    extra_room = np.random.uniform(-0.05, +0.05, 2) 
                    if grid[landmark_location_grid[0],landmark_location_grid[1]]==1:
                        continue
                    else:
                        grid[landmark_location_grid[0],landmark_location_grid[1]] = 1
                        one_starts_landmark_grid.append(copy.deepcopy(landmark_location_grid))
                        landmark_location = np.array([(landmark_location_grid[0]+0.5)*cell_size,(landmark_location_grid[1]+0.5)*cell_size]) + extra_room -start_boundary
                        one_starts_landmark.append(copy.deepcopy(landmark_location))
                        break
            indices = random.sample(range(now_agent_num), now_agent_num)
            for k in indices:
                epsilons = np.array([[-1,0],[1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
                epsilon = epsilons[random.sample(range(8),8)]
                # extra_room = -2 * 0.02 * random.random() + 0.02
                for epsilon_id in range(epsilon.shape[0]):
                    agent_location_grid = one_starts_landmark_grid[k] + epsilon[epsilon_id]
                    if agent_location_grid[0] >= grid.shape[0]:
                        agent_location_grid[0] = grid.shape[0]-1
                    if agent_location_grid[1] >= grid.shape[1]:
                        agent_location_grid[1] = grid.shape[1]-1
                    if grid[agent_location_grid[0],agent_location_grid[1]]!=2:
                        grid[agent_location_grid[0],agent_location_grid[1]]=2
                        break
                noise = np.random.uniform(-0.01, +0.01)
                agent_location = np.array([(agent_location_grid[0]+0.5)*cell_size,(agent_location_grid[1]+0.5)*cell_size])-start_boundary+noise
                one_starts_agent.append(copy.deepcopy(agent_location))
            # select_starts.append(one_starts_agent+one_starts_landmark)
            archive.append(one_starts_agent+one_starts_landmark)
            grid = np.zeros(shape=(grid_num,grid_num))
            one_starts_agent = []
            one_starts_landmark_grid = []
            one_starts_landmark = []
        return archive

    def get_novelty(self,list1,list2):
        # list1是需要求novelty的
        topk=5
        dist = cdist(np.array(list1).reshape(len(list1),-1),np.array(list2).reshape(len(list2),-1),metric='euclidean')
        if len(list2) < topk+1:
            dist_k = dist
            novelty = np.sum(dist_k,axis=1)/len(list2)
        else:
            dist_k = np.partition(dist,topk+1,axis=1)[:,0:topk+1]
            novelty = np.sum(dist_k,axis=1)/topk
        return novelty

    def novelty_sort(self, buffer, buffer_novelty):
        zipped = zip(buffer,buffer_novelty)
        sort_zipped = sorted(zipped,key=lambda x:(x[1],np.mean(x[0])))
        result = zip(*sort_zipped)
        buffer_new, buffer_novelty_new = [list(x) for x in result]
        return buffer_new, buffer_novelty_new

    def SampleNearby_novelty(self, parents, writer, timestep): # produce high novelty children and return 
        if len(self.parent_all) > self.topk + 1:
            self.parent_all_novelty = self.get_novelty(self.parent_all,self.parent_all)
            self.parent_all, self.parent_all_novelty = self.novelty_sort(self.parent_all, self.parent_all_novelty)
            novelty_threshold = np.mean(self.parent_all_novelty)
        else:
            novelty_threshold = 0
        wandb.log({str(self.num_agents)+'novelty_threshold': novelty_threshold},timestep)
        parents = parents + []
        len_start = len(parents)
        child_new = []
        if parents==[]:
            return []
        else:
            add_num = 0
            while add_num < self.reproduction_num:
                for k in range(len_start):
                    st = copy.deepcopy(parents[k])
                    s_len = len(st)
                    for i in range(s_len):
                        epsilon_x = -2 * self.max_step * random.random() + self.max_step
                        epsilon_y = -2 * self.max_step * random.random() + self.max_step
                        st[i][0] = st[i][0] + epsilon_x
                        st[i][1] = st[i][1] + epsilon_y
                        if st[i][0] > self.boundary:
                            st[i][0] = self.boundary - random.random()*0.01
                        if st[i][0] < -self.boundary:
                            st[i][0] = -self.boundary + random.random()*0.01
                        if st[i][1] > self.boundary:
                            st[i][1] = self.boundary - random.random()*0.01
                        if st[i][1] < -self.boundary:
                            st[i][1] = -self.boundary + random.random()*0.01
                    if len(self.parent_all) > self.topk + 1:
                        if self.get_novelty([st],self.parent_all) > novelty_threshold:
                            child_new.append(copy.deepcopy(st))
                            add_num += 1
                    else:
                        child_new.append(copy.deepcopy(st))
                        add_num += 1
            child_new = random.sample(child_new, min(self.reproduction_num,len(child_new)))
            return child_new

    def SampleNearby(self, starts): # produce new children and return
        starts = starts + []
        len_start = len(starts)
        starts_new = []
        if starts==[]:
            return []
        else:
            add_num = 0
            while add_num < self.reproduction_num:
                for i in range(len_start):
                    st = copy.deepcopy(starts[i])
                    s_len = len(st)
                    for i in range(s_len):
                        epsilon_x = -2 * self.max_step * random.random() + self.max_step
                        epsilon_y = -2 * self.max_step * random.random() + self.max_step
                        st[i][0] = st[i][0] + epsilon_x
                        st[i][1] = st[i][1] + epsilon_y
                        if st[i][0] > self.boundary:
                            st[i][0] = self.boundary - random.random()*0.01
                        if st[i][0] < -self.boundary:
                            st[i][0] = -self.boundary + random.random()*0.01
                        if st[i][1] > self.boundary:
                            st[i][1] = self.boundary - random.random()*0.01
                        if st[i][1] < -self.boundary:
                            st[i][1] = -self.boundary + random.random()*0.01
                    starts_new.append(copy.deepcopy(st))
                    add_num += 1
            starts_new = random.sample(starts_new, self.reproduction_num)
            return starts_new

    def sample_starts(self, N_child, N_archive, N_parent=0):
        self.choose_child_index = random.sample(range(len(self.childlist)), min(len(self.childlist), N_child))
        self.choose_parent_index = random.sample(range(len(self.parent_all)),min(len(self.parent_all), N_parent))
        self.choose_archive_index = random.sample(range(len(self.archive)), min(len(self.archive), N_child + N_archive + N_parent - len(self.choose_child_index)-len(self.choose_parent_index)))
        if len(self.choose_archive_index) < N_archive:
            self.choose_child_index = random.sample(range(len(self.childlist)), min(len(self.childlist), N_child + N_archive + N_parent - len(self.choose_archive_index)-len(self.choose_parent_index)))
        if len(self.choose_child_index) < N_child:
            self.choose_parent_index = random.sample(range(len(self.parent_all)), min(len(self.parent_all), N_child + N_archive + N_parent - len(self.choose_archive_index)-len(self.choose_child_index)))
        # fix the thread problem
        tmp_index_length = len(self.choose_archive_index) + len(self.choose_child_index) + len(self.choose_parent_index)
        sum_N = N_child + N_archive + N_parent
        if  tmp_index_length < sum_N:
            self.choose_parent_index += random.sample(range(len(self.parent_all)),min(len(self.parent_all), sum_N - tmp_index_length))
        self.choose_child_index = np.sort(self.choose_child_index)
        self.choose_archive_index = np.sort(self.choose_archive_index)
        self.choose_parent_index = np.sort(self.choose_parent_index)
        one_length = len(self.choose_child_index) + len(self.choose_archive_index) # 需要搬运的点个数
        starts_length = len(self.choose_child_index) + len(self.choose_archive_index) + len(self.choose_parent_index)
        starts = []
        for i in range(len(self.choose_child_index)):
            starts.append(self.childlist[self.choose_child_index[i]])
        for i in range(len(self.choose_archive_index)):
            starts.append(self.archive[self.choose_archive_index[i]])
        for i in range(len(self.choose_parent_index)):
            starts.append(self.parent_all[self.choose_parent_index[i]])
        # print('sample_archive: ', len(self.choose_archive_index))
        # print('sample_childlist: ', len(self.choose_child_index))
        # print('sample_parent: ', len(self.choose_parent_index))
        # print('%isample_length: '%self.num_agents, starts_length)
        return starts, one_length, starts_length
    
    def move_nodes(self, one_length, Rmax, Rmin, del_switch, writer, timestep): 
        del_child_num = 0
        del_archive_num = 0
        del_easy_num = 0
        add_hard_num = 0
        drop_num = 0
        self.parent = []
        child2archive = []
        for i in range(one_length):
            if i < len(self.choose_child_index):
                if self.eval_score[i]>=Rmin and self.eval_score[i]<=Rmax:
                    child2archive.append(copy.deepcopy(self.childlist[self.choose_child_index[i]-del_child_num]))
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
                elif self.eval_score[i] > Rmax:
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
                    drop_num += 1
                else:
                    self.hardlist.append(copy.deepcopy(self.childlist[self.choose_child_index[i]-del_child_num]))
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
                    add_hard_num += 1
            else:
                if self.eval_score[i]>Rmax:
                    self.parent.append(copy.deepcopy(self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]))
                    del self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]
                    del_archive_num += 1
        self.archive += child2archive
        self.parent_all += self.parent
        # print('child_drop: ', drop_num)
        # print('add_hard_num: ', add_hard_num )
        # print('parent: ', len(self.parent))
        if len(self.childlist) > self.buffer_length:
            self.childlist = self.childlist[len(self.childlist)-self.buffer_length:]
        if len(self.archive) > self.buffer_length:
            if del_switch=='novelty' : # novelty del
                self.archive_novelty = self.get_novelty(self.archive,self.archive)
                self.archive,self.archive_novelty = self.novelty_sort(self.archive,self.archive_novelty)
                self.archive = self.archive[len(self.archive)-self.buffer_length:]
            elif del_switch=='random': # random del
                del_num = len(self.archive) - self.buffer_length
                del_index = random.sample(range(len(self.archive)),del_num)
                del_index = np.sort(del_index)
                del_archive_num = 0
                for i in range(del_num):
                    del self.archive[del_index[i]-del_archive_num]
                    del_archive_num += 1
            else: # old del
                self.archive = self.archive[len(self.archive)-self.buffer_length:]
        if len(self.parent_all) > self.buffer_length:
            self.parent_all = self.parent_all[len(self.parent_all)-self.buffer_length:]
        wandb.log({str(self.num_agents)+'archive_length': len(self.archive)},timestep)
        wandb.log({str(self.num_agents)+'childlist_length': len(self.childlist)},timestep)
        wandb.log({str(self.num_agents)+'parentlist_length': len(self.parent)},timestep)
        wandb.log({str(self.num_agents)+'drop_num': drop_num},timestep)
    
    def save_node(self, dir_path, episode):
        # dir_path: '/home/chenjy/mappo-curriculum/' + args.model_dir
        if self.num_agents!=0:
            save_path = dir_path / ('%iagents' % (self.num_agents))
            if not os.path.exists(save_path):
                os.makedirs(save_path / 'childlist')
                os.makedirs(save_path / 'archive')
                os.makedirs(save_path / 'archive_novelty')
                os.makedirs(save_path / 'parent')
                os.makedirs(save_path / 'parent_all')
            with open(save_path / 'childlist'/ ('child_%i' %(episode)),'w+') as fp:
                for line in self.childlist:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'archive' / ('archive_%i' %(episode)),'w+') as fp:
                for line in self.archive:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            self.novelty = self.get_novelty(self.archive,self.archive)
            with open(save_path / 'archive_novelty' / ('archive_novelty_%i' %(episode)),'w+') as fp:
                for line in self.archive_novelty:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'parent' / ('parent_%i' %(episode)),'w+') as fp:
                for line in self.parent:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'parent_all' / ('parent_all_%i' %(episode)),'w+') as fp:
                for line in self.parent_all:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
        else:
            return 0

    def save_phase_curricula(self, dir_path, success_rate):
        if self.num_agents!=0:
            save_path = dir_path / ('%iagents' % (self.num_agents))
            if not os.path.exists(save_path):
                os.makedirs(save_path / 'archive')
                os.makedirs(save_path / 'parent')
                os.makedirs(save_path / 'parent_all')
            with open(save_path / 'archive' / ('archive_%f' %(success_rate)),'w+') as fp:
                for line in self.archive:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'parent' / ('parent_%f' %(success_rate)),'w+') as fp:
                for line in self.parent:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / 'parent_all' / ('parent_all_%f' %(success_rate)),'w+') as fp:
                for line in self.parent_all:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
        else:
            return 0

def evaluation(envs, actor_critic, args, eval_num_agents, timestep):
    # update envs
    envs.close()
    args.n_rollout_threads = 500
    envs = make_parallel_env(args)
    # reset num_agents
    actor_critic.num_agents = eval_num_agents
    obs, _ = envs.reset(eval_num_agents)
    #replay buffer
    rollouts = RolloutStorage_share(eval_num_agents,
                args.test_episode_length, 
                args.n_rollout_threads,
                envs.observation_space[0], 
                envs.action_space[0],
                args.hidden_size) 
    # replay buffer init
    if args.share_policy: 
        share_obs = obs.reshape(args.n_rollout_threads, -1)        
        # share_obs = np.expand_dims(share_obs,1).repeat(now_node.agent_num,axis=1)    
        rollouts.share_obs[0] = share_obs.copy() 
        rollouts.obs[0] = obs.copy()               
        rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
        rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    else:
        share_obs = []
        for o in obs:
            share_obs.append(list(itertools.chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(eval_num_agents):    
            rollouts[agent_id].share_obs[0] = share_obs.copy()
            rollouts[agent_id].obs[0] = np.array(list(obs[:,agent_id])).copy()               
            rollouts[agent_id].recurrent_hidden_states = np.zeros(rollouts[agent_id].recurrent_hidden_states.shape).astype(np.float32)
            rollouts[agent_id].recurrent_hidden_states_critic = np.zeros(rollouts[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)
    test_cover_rate = np.zeros(shape=(args.n_rollout_threads,args.test_episode_length))
    test_collision_num = np.zeros(shape=args.n_rollout_threads)
    test_success = np.zeros(shape=(args.n_rollout_threads,args.test_episode_length))
    for step in range(args.test_episode_length):
        # Sample actions
        values = []
        actions= []
        action_log_probs = []
        recurrent_hidden_statess = []
        recurrent_hidden_statess_critic = []
        
        with torch.no_grad():                
            for agent_id in range(eval_num_agents):
                if args.share_policy:
                    actor_critic.eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                        # torch.FloatTensor(rollouts.share_obs[step,:,agent_id]), 
                        torch.FloatTensor(rollouts.share_obs[step]), 
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
            for agent_id in range(eval_num_agents):
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
        obs, rewards, dones, infos, _ = envs.step(actions_env, args.n_rollout_threads, eval_num_agents)
        cover_rate_list = []
        collision_list = []
        success_list = []
        for env_id in range(args.n_rollout_threads):
            cover_rate_list.append(infos[env_id][0]['cover_rate'])
            collision_list.append(infos[env_id][0]['collision'])
            success_list.append(int(infos[env_id][0]['success']))
        test_cover_rate[:,step] = np.array(cover_rate_list)
        test_collision_num += np.array(collision_list)
        test_success[:,step] = np.array(success_list)
        # step_cover_rate[:,step] = np.array(infos)[:,0]

        # If done then clean the history of observations.
        # insert data in buffer
        masks = []
        for i, done in enumerate(dones): 
            mask = []               
            for agent_id in range(eval_num_agents): 
                if done[agent_id]:    
                    recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                    recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                    mask.append([0.0])
                else:
                    mask.append([1.0])
            masks.append(mask)
                        
        if args.share_policy: 
            share_obs = obs.reshape(args.n_rollout_threads, -1)        
            # share_obs = np.expand_dims(share_obs,1).repeat(now_node.agent_num,axis=1)    
            
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
            for agent_id in range(eval_num_agents):
                rollouts[agent_id].insert(share_obs, 
                        np.array(list(obs[:,agent_id])), 
                        np.array(recurrent_hidden_statess[agent_id]), 
                        np.array(recurrent_hidden_statess_critic[agent_id]), 
                        np.array(actions[agent_id]),
                        np.array(action_log_probs[agent_id]), 
                        np.array(values[agent_id]),
                        rewards[:,agent_id], 
                        np.array(masks)[:,agent_id])
    mean_cover_rate = np.mean(np.mean(test_cover_rate[:,-args.historical_length:],axis=1))
    mean_success_rate = np.mean(np.mean(test_success[:,-args.historical_length:],axis=1))
    collision_num = np.mean(test_collision_num)
    rew = []
    for i in range(rollouts.rewards.shape[1]):
        rew.append(np.sum(rollouts.rewards[:,i]))
    average_episode_reward = np.mean(rew)
    envs.close()
    return mean_cover_rate, mean_success_rate, collision_num, average_episode_reward

def collect_data(envs, agents, actor_critic, args, node, starts, starts_length, one_length, timestep):
    # update envs
    envs.close()
    args.n_rollout_threads = starts_length
    envs = make_parallel_env(args)
    # reset num_agents
    actor_critic.num_agents = node.num_agents
    obs = envs.new_starts_obs(starts, node.num_agents, starts_length)
    #replay buffer
    rollouts = RolloutStorage_share(node.num_agents,
                args.episode_length, 
                starts_length,
                envs.observation_space[0], 
                envs.action_space[0],
                args.hidden_size) 
    # replay buffer init
    if args.share_policy: 
        share_obs = obs.reshape(starts_length, -1)   
        rollouts.share_obs[0] = share_obs.copy() 
        rollouts.obs[0] = obs.copy()               
        rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
        rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    else:
        share_obs = []
        for o in obs:
            share_obs.append(list(itertools.chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(node.num_agents):    
            rollouts[agent_id].share_obs[0] = share_obs.copy()
            rollouts[agent_id].obs[0] = np.array(list(obs[:,agent_id])).copy()               
            rollouts[agent_id].recurrent_hidden_states = np.zeros(rollouts[agent_id].recurrent_hidden_states.shape).astype(np.float32)
            rollouts[agent_id].recurrent_hidden_states_critic = np.zeros(rollouts[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)
    step_cover_rate = np.zeros(shape=(one_length,args.episode_length))
    step_collision_num = np.zeros(shape=one_length)
    step_success = np.zeros(shape=(one_length,args.episode_length))
    for step in range(args.episode_length):
        # Sample actions
        values = []
        actions= []
        action_log_probs = []
        recurrent_hidden_statess = []
        recurrent_hidden_statess_critic = []
        
        with torch.no_grad():                
            for agent_id in range(node.num_agents):
                if args.share_policy:
                    actor_critic.eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
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
        for i in range(starts_length):
            one_hot_action_env = []
            for agent_id in range(node.num_agents):
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
        obs, rewards, dones, infos, _ = envs.step(actions_env, starts_length, node.num_agents)
        cover_rate_list = []
        collision_list = []
        success_list = []
        for env_id in range(one_length):
            cover_rate_list.append(infos[env_id][0]['cover_rate'])
            collision_list.append(infos[env_id][0]['collision'])
            success_list.append(int(infos[env_id][0]['success']))
        step_cover_rate[:,step] = np.array(cover_rate_list)
        step_collision_num += np.array(collision_list)
        step_success[:,step] = np.array(success_list)

        # If done then clean the history of observations.
        # insert data in buffer
        masks = []
        for i, done in enumerate(dones): 
            mask = []               
            for agent_id in range(node.num_agents): 
                if done[agent_id]:    
                    recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                    recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                    mask.append([0.0])
                else:
                    mask.append([1.0])
            masks.append(mask)
                        
        if args.share_policy: 
            share_obs = obs.reshape(starts_length, -1)           
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
            for agent_id in range(node.num_agents):
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
    wandb.log({str(node.num_agents)+'training_cover_rate': np.mean(np.mean(step_cover_rate[:,-args.historical_length:],axis=1))}, timestep)
    wandb.log({str(node.num_agents)+'training_success_rate': np.mean(np.mean(step_success[:,-args.historical_length:],axis=1))}, timestep)
    print(str(node.num_agents) + 'training_cover_rate: ', np.mean(np.mean(step_cover_rate[:,-args.historical_length:],axis=1)), end=' ')
    print('threads: ', args.n_rollout_threads)
    wandb.log({str(node.num_agents)+'train_collision_num': np.mean(step_collision_num)},timestep)
    timestep += args.episode_length * starts_length
    node.eval_score += np.mean(step_cover_rate[:,-args.historical_length:],axis=1)
                                
    with torch.no_grad():  # get value and compute return
        for agent_id in range(node.num_agents):         
            if args.share_policy: 
                actor_critic.eval()                
                next_value,_,_ = actor_critic.get_value(agent_id,
                                            # torch.FloatTensor(rollouts_last.share_obs[-1,:,agent_id]), 
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
    envs.close()
    return rollouts, timestep

def save(save_path, agents):
    torch.save({'model': agents.actor_critic, 
                'optimizer_actor': agents.optimizer_actor,
                'optimizer_critic': agents.optimizer_critic}, save_path)


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
    boundary = 3
    start_boundary = 0.3
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
        stop_mix_signal = upper_bound
        mix_add_frequency = 1 # 改变比例的频率
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
                           boundary=boundary)
    now_node = node_buffer(now_agent_num,buffer_length,
                           archive_initial_length=int(args.n_rollout_threads),
                           reproduction_num=M,
                           max_step=max_step,
                           start_boundary=start_boundary,
                           boundary=boundary)
    
    # region load curricula and model
    load_curricula = True
    initial_optimizer = False
    warm_up = False
    warmup_iter = 30
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
        start_boundary = 1.0
        if not transfer:
            mix_flag = True
        # initialize now node
        now_node = node_buffer(now_agent_num,buffer_length,
                        archive_initial_length=int(args.n_rollout_threads),
                        reproduction_num=M,
                        max_step=max_step,
                        start_boundary=start_boundary,
                        boundary=boundary)
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
                        value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(now_node.num_agents, rollouts_now, warm_up=warm_up)
                    else:
                        if last_node.num_agents!=0 and mix_flag:
                            wandb.log({'Type of agents': 2}, current_timestep)
                            value_loss, action_loss, dist_entropy = agents.update_double_share(last_node.num_agents, now_node.num_agents, rollouts_last, rollouts_now)
                            # clean the buffer and reset
                            rollouts_last.after_update()
                        else:
                            wandb.log({'Type of agents': 1},current_timestep)
                            value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(now_node.num_agents, rollouts_now, warm_up=False)
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
        if mean_cover_rate_current >= upper_bound:
            entity_count += 1
        if entity_count >= eval_frequency and now_node.num_agents < target_num:
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
                                boundary=boundary)
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