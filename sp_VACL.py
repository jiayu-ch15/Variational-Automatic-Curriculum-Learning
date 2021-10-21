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
import matplotlib.pyplot as plt
import pdb
import wandb
np.set_printoptions(linewidth=1000)

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

def log_infos(args, infos, timestep, logger=None):
    if args.use_wandb:
        for keys, values in infos.items():
            wandb.log({keys: values},step=timestep)
    else:
        for keys,values in infos.items():
            logger.add_scalars(keys, {keys: values}, timestep)

class node_buffer():
    def __init__(self, args, num_agents, buffer_length,archive_initial_length,reproduction_num,epsilon,delta):
        self.env_name = args.env_name
        self.scenario_name = args.scenario_name
        self.num_agents = num_agents
        self.buffer_length = buffer_length
        self.buffer_min_length = archive_initial_length
        self.archive = self.initial_tasks(archive_initial_length, self.num_agents)
        self.archive_score = np.zeros(len(self.archive))
        self.archive_novelty = self.get_novelty(self.archive,self.archive)
        self.archive, self.archive_novelty = self.novelty_sort(self.archive, self.archive_novelty)
        self.childlist = []
        self.hardlist = []
        self.parent = []
        self.parent_all = []
        if self.env_name == 'MPE' and self.scenario_name == 'simple_spread':
            self.legal_region = {'agent':{'x':[[-3,3]],'y': [[-3,3]]}, 'landmark':{'x':[[-3,3]],'y': [[-3,3]]}}
        elif self.env_name == 'MPE' and self.scenario_name == 'push_ball':
            self.legal_region = {'agent':{'x':[[-2,2]],'y': [[-2,2]]}, 'landmark':{'x':[[-2,2]],'y': [[-2,2]]}}
        elif self.env_name == 'MPE' and self.scenario_name == 'hard_spread':
            self.legal_region = {'agent':{'x':[[-4.9,-3.1],[-3,-1],[-0.9,0.9],[1,3],[3.1,4.9]],
                                 'y': [[-0.9,0.9],[0.45,0.75],[-0.9,0.9],[-0.75,-0.45],[-0.9,0.9]]},
                                'landmark':{'x':[[3.1,4.9]],'y':[[-0.9,0.9]]}} 
        self.reproduction_num = reproduction_num
        self.choose_child_index = []
        self.choose_archive_index = []
        self.eval_score = np.zeros(shape=len(self.archive))
        self.topk = 5
        self.epsilon = epsilon
        self.delta = delta

    def initial_tasks(self, num_case, num_agents):
        if self.env_name == 'MPE' and self.scenario_name == 'simple_spread':
            if num_agents <= 4:
                start_boundary = [-0.3,0.3,-0.3,0.3]
            else:
                start_boundary = [-1.0,1.0,-1.0,1.0]
            one_starts_landmark = []
            one_starts_agent = []
            archive = [] 
            for j in range(num_case):
                for i in range(num_agents):
                    landmark_location = np.array([np.random.uniform(start_boundary[0],start_boundary[1]),np.random.uniform(start_boundary[2],start_boundary[3])])
                    one_starts_landmark.append(copy.deepcopy(landmark_location))
                indices = random.sample(range(num_agents), num_agents)
                for k in indices:
                    epsilon = -2 * 0.01 * random.random() + 0.01
                    one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
                archive.append(one_starts_agent+one_starts_landmark)
                one_starts_agent = []
                one_starts_landmark = []
            return archive
        elif self.env_name == 'MPE' and self.scenario_name == 'push_ball':
            landmark_size = 0.3
            box_size = 0.3
            agent_size = 0.2
            if num_agents <= 2:
                start_boundary = [-0.4,0.4,-0.4,0.4]
            else:
                start_boundary = [-0.8,0.8,-0.8,0.8]
            num_boxes = num_agents
            cell_size = max([landmark_size,box_size,agent_size]) + 0.1
            grid_num = round((start_boundary[1]-start_boundary[0]) / cell_size)
            init_origin_node = np.array([start_boundary[0]+0.5*cell_size,start_boundary[3]-0.5*cell_size]) # left, up
            assert grid_num ** 2 >= num_agents + num_boxes
            grid = np.zeros(shape=(grid_num,grid_num))
            grid_without_landmark = np.zeros(shape=(grid_num,grid_num))
            one_starts_landmark = []
            one_starts_agent = []
            one_starts_box = []
            one_starts_box_grid = []
            archive = [] 
            for j in range(num_case):
                # box location
                for i in range(num_boxes):
                    while 1:
                        box_location_grid = np.random.randint(0, grid.shape[0], 2) 
                        if grid[box_location_grid[0],box_location_grid[1]]==1:
                            continue
                        else:
                            grid[box_location_grid[0],box_location_grid[1]] = 1
                            extra_room = (cell_size - landmark_size) / 2
                            extra_x = np.random.uniform(-extra_room,extra_room)
                            extra_y = np.random.uniform(-extra_room,extra_room)
                            box_location = np.array([(box_location_grid[0]+0.5)*cell_size+extra_x,-(box_location_grid[1]+0.5)*cell_size+extra_y]) + init_origin_node
                            one_starts_box.append(copy.deepcopy(box_location))
                            one_starts_box_grid.append(copy.deepcopy(box_location_grid))
                            break
                grid_without_landmark = copy.deepcopy(grid)
                # landmark location
                indices = random.sample(range(num_boxes), num_boxes)
                num_try = 0
                num_tries = 50
                for k in indices:
                    around = 1
                    while num_try < num_tries:
                        delta_x_direction = random.randint(-around,around)
                        delta_y_direction = random.randint(-around,around)
                        landmark_location_x = min(max(0,one_starts_box_grid[k][0]+delta_x_direction),grid.shape[0]-1)
                        landmark_location_y = min(max(0,one_starts_box_grid[k][1]+delta_y_direction),grid.shape[1]-1)
                        landmark_location_grid = np.array([landmark_location_x,landmark_location_y])
                        if grid[landmark_location_grid[0],landmark_location_grid[1]]==1:
                            num_try += 1
                            if num_try >= num_tries and around==1:
                                around = 2
                                num_try = 0
                            assert num_try<num_tries or around==1, 'case %i can not find landmark pos'%j
                            continue
                        else:
                            grid[landmark_location_grid[0],landmark_location_grid[1]] = 1
                            extra_room = (cell_size - landmark_size) / 2
                            extra_x = np.random.uniform(-extra_room,extra_room)
                            extra_y = np.random.uniform(-extra_room,extra_room)
                            landmark_location = np.array([(landmark_location_grid[0]+0.5)*cell_size+extra_x,-(landmark_location_grid[1]+0.5)*cell_size+extra_y]) + init_origin_node
                            one_starts_landmark.append(copy.deepcopy(landmark_location))
                            break
                # agent_location
                indices_agent = random.sample(range(num_boxes), num_boxes)
                num_try = 0
                num_tries = 50
                for k in indices_agent:
                    around = 1
                    while num_try < num_tries:
                        delta_x_direction = random.randint(-around,around)
                        delta_y_direction = random.randint(-around,around)
                        agent_location_x = one_starts_box_grid[k][0]+delta_x_direction
                        agent_location_y = one_starts_box_grid[k][1]+delta_y_direction
                        agent_location_grid = np.array([agent_location_x,agent_location_y])
                        if agent_location_x < 0 or agent_location_y < 0 or agent_location_x > grid.shape[0]-1 or  agent_location_y > grid.shape[0]-1:
                            extra_room = (cell_size - landmark_size) / 2
                            extra_x = np.random.uniform(-extra_room,extra_room)
                            extra_y = np.random.uniform(-extra_room,extra_room)
                            agent_location = np.array([(agent_location_grid[0]+0.5)*cell_size+extra_x,-(agent_location_grid[1]+0.5)*cell_size+extra_y]) + init_origin_node
                            one_starts_agent.append(copy.deepcopy(agent_location))
                            break
                        else:
                            if grid_without_landmark[agent_location_grid[0],agent_location_grid[1]]==1:
                                num_try += 1
                                if num_try >= num_tries and around==1:
                                    around = 2
                                    num_try = 0
                                assert num_try<num_tries or around==1, 'case %i can not find agent pos'%j
                                continue
                            else:
                                grid_without_landmark[agent_location_grid[0],agent_location_grid[1]] = 1
                                extra_room = (cell_size - landmark_size) / 2
                                extra_x = np.random.uniform(-extra_room,extra_room)
                                extra_y = np.random.uniform(-extra_room,extra_room)
                                agent_location = np.array([(agent_location_grid[0]+0.5)*cell_size+extra_x,-(agent_location_grid[1]+0.5)*cell_size+extra_y]) + init_origin_node
                                one_starts_agent.append(copy.deepcopy(agent_location))
                                break
                # select_starts.append(one_starts_agent+one_starts_landmark)
                archive.append(one_starts_agent+one_starts_box+one_starts_landmark)
                grid = np.zeros(shape=(grid_num,grid_num))
                grid_without_landmark = np.zeros(shape=(grid_num,grid_num))
                one_starts_agent = []
                one_starts_landmark = []
                one_starts_box = []
                one_starts_box_grid = []
            return archive
        elif self.env_name == 'MPE' and self.scenario_name == 'hard_spread':
            start_boundary = {'x':[[3.7,4.3]],'y':[[-0.3,0.3]]} # good goal
            one_starts_landmark = []
            one_starts_agent = []
            archive = [] 
            # easy goal是agent和landmark都在left and right
            start_boundary_x = start_boundary['x']
            start_boundary_y = start_boundary['y']
            for j in range(num_case):
                for i in range(num_agents):
                    location_id = np.random.randint(len(start_boundary_x))
                    landmark_location_x = np.random.uniform(start_boundary_x[location_id][0],start_boundary_x[location_id][1])
                    landmark_location_y = np.random.uniform(start_boundary_y[location_id][0],start_boundary_y[location_id][1])
                    landmark_location = np.array([landmark_location_x,landmark_location_y])
                    one_starts_landmark.append(copy.deepcopy(landmark_location))
                indices = random.sample(range(num_agents), num_agents)
                for k in indices:
                    epsilon = -2 * 0.01 * random.random() + 0.01
                    one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
                archive.append(one_starts_agent+one_starts_landmark)
                one_starts_agent = []
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
    
    def Sample_gradient(self,parents,timestep,h=100.0, use_gradient_noise=True):
        boundary_x_agent = self.legal_region['agent']['x']
        boundary_y_agent = self.legal_region['agent']['y']
        boundary_x_landmark = self.legal_region['landmark']['x']
        boundary_y_landmark = self.legal_region['landmark']['y']
        parents = parents + []
        len_start = len(parents)
        child_new = []
        if parents==[]:
            return []
        else:
            add_num = 0
            while add_num < self.reproduction_num:
                for parent in parents:
                    parent_gradient, parent_gradient_zero = self.gradient_of_state(np.array(parent).reshape(-1),self.parent_all,h=h)
                    
                    # gradient step
                    new_parent = []
                    for parent_of_entity_id in range(len(parent)):
                        st = copy.deepcopy(parent[parent_of_entity_id])
                        # execute gradient step
                        if not parent_gradient_zero:
                            st[0] += parent_gradient[parent_of_entity_id * 2] * self.epsilon
                            st[1] += parent_gradient[parent_of_entity_id * 2 + 1] * self.epsilon
                        else:
                            stepsizex = -2 * self.epsilon * random.random() + self.epsilon
                            stepsizey = -2 * self.epsilon * random.random() + self.epsilon
                            st[0] += stepsizex
                            st[1] += stepsizey
                        # clip
                        if parent_of_entity_id < self.num_agents:
                            boundary_x = boundary_x_agent
                            boundary_y = boundary_y_agent
                        else:
                            boundary_x = boundary_x_landmark
                            boundary_y = boundary_y_landmark
                        st = self.clip_states(st,boundary_x,boundary_y)
                        # rejection sampling
                        if use_gradient_noise:
                            num_tries = 100
                            num_try = 0
                            while num_try <= num_tries:
                                epsilon_x = -2 * self.delta * random.random() + self.delta
                                epsilon_y = -2 * self.delta * random.random() + self.delta
                                tmp_x = st[0] + epsilon_x
                                tmp_y = st[1] + epsilon_y
                                is_legal = self.is_legal([tmp_x,tmp_y],boundary_x,boundary_y)
                                num_try += 1
                                if is_legal:
                                    st[0] = tmp_x
                                    st[1] = tmp_y
                                    break
                                else:
                                    assert num_try <= num_tries, str(st)
                                    continue
                        new_parent.append(st)
                    child_new.append(new_parent)
                    add_num += 1
                    if add_num >= self.reproduction_num: break
            return child_new

    def gradient_of_state(self,state,buffer,h=100.0,use_rbf=True):
        gradient = np.zeros(state.shape)
        for buffer_state in buffer:
            if use_rbf:
                dist0 = state - np.array(buffer_state).reshape(-1)
                # gradient += -2 * dist0 * np.exp(-dist0**2 / h) / h
                gradient += 2 * dist0 * np.exp(-dist0**2 / h) / h
            else:
                gradient += 2 * (state - np.array(buffer_state).reshape(-1))
        norm = np.linalg.norm(gradient, ord=2)
        if norm > 0.0:
            gradient = gradient / np.linalg.norm(gradient, ord=2)
            gradient_zero = False
        else:
            gradient_zero = True
        return gradient, gradient_zero

    def is_legal(self, pos, boundary_x, boundary_y):
        legal = False
        # 限制在整个大的范围内
        if pos[0] < boundary_x[0][0] or pos[0] > boundary_x[-1][1]:
            return False
        # boundary_x = [[-4.9,-3.1],[-3,-1],[-0.9,0.9],[1,3],[3.1,4.9]]
        for boundary_id in range(len(boundary_x)):
            if pos[0] >= boundary_x[boundary_id][0] and pos[0] <= boundary_x[boundary_id][1]:
                if pos[1] >= boundary_y[boundary_id][0] and pos[1] <= boundary_y[boundary_id][1]:
                    legal = True
                    break
        return legal

    def clip_states(self,pos, boundary_x, boundary_y):
        # boundary_x = [[-4.9,-3.1],[-3,-1],[-0.9,0.9],[1,3],[3.1,4.9]]
        # clip to [-map,map]
        if pos[0] < boundary_x[0][0]:
            pos[0] = boundary_x[0][0] + random.random()*0.01
        elif pos[0] > boundary_x[-1][1]:
            pos[0] = boundary_x[-1][1] - random.random()*0.01

        for boundary_id in range(len(boundary_x)):
            if pos[0] >= boundary_x[boundary_id][0] and pos[0] <= boundary_x[boundary_id][1]:
                if pos[1] >= boundary_y[boundary_id][0] and pos[1] <= boundary_y[boundary_id][1]:
                    break
                elif pos[1] < boundary_y[boundary_id][0]:
                    pos[1] = boundary_y[boundary_id][0] + random.random()*0.01
                elif pos[1] > boundary_y[boundary_id][1]:
                    pos[1] = boundary_y[boundary_id][1] - random.random()*0.01
        return pos

    def sample_starts(self, N_archive, N_sol):
        self.choose_parent_index = random.sample(range(len(self.parent_all)),min(len(self.parent_all), N_sol))
        self.choose_archive_index = random.sample(range(len(self.archive)), min(len(self.archive), N_archive + N_sol - len(self.choose_parent_index)))
        if len(self.choose_archive_index) < N_archive:
            self.choose_parent_index = random.sample(range(len(self.parent_all)), min(len(self.parent_all), N_archive + N_sol - len(self.choose_archive_index)))
        self.choose_archive_index = np.sort(self.choose_archive_index)
        self.choose_parent_index = np.sort(self.choose_parent_index)
        active_batch = len(self.choose_archive_index)
        all_batch = len(self.choose_archive_index) + len(self.choose_parent_index)
        starts = []
        for i in range(len(self.choose_archive_index)):
            starts.append(self.archive[self.choose_archive_index[i]])
        for i in range(len(self.choose_parent_index)):
            starts.append(self.parent_all[self.choose_parent_index[i]])
        return starts, active_batch, all_batch

    def update_buffer(self, active_batch, Rmax, Rmin, del_switch, timestep):
        del_archive_num = 0
        del_easy_num = 0
        add_hard_num = 0
        self.parent = []
        for i in range(active_batch):
            if self.eval_score[i] > Rmax:
                self.parent.append(copy.deepcopy(self.archive[self.choose_archive_index[i]-del_archive_num]))
                del self.archive[self.choose_archive_index[i]-del_archive_num]
                del_archive_num += 1
            elif self.eval_score[i] < Rmin:
                if len(self.archive) > self.buffer_min_length:
                    del self.archive[self.choose_archive_index[i]-del_archive_num]
                    del_archive_num += 1
        self.parent_all += self.parent
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
        return len(self.archive), len(self.parent), del_archive_num

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
            if len(self.archive) > 0:
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

def main():
    args = get_config()
    if args.use_wandb:
        run = wandb.init(project='curriculum',
                         name=str(args.algorithm_name) + "_seed" + str(args.seed))
    
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
        actor_base = ATTBase_actor(envs.observation_space[0].shape[0], envs.action_space[0], num_agents, args.scenario_name)
        critic_base = ATTBase_critic(envs.observation_space[0].shape[0], num_agents, args.scenario_name)
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
    
    # set parameters from config
    N_sol = int(args.n_rollout_threads * args.sol_prop)
    N_archive = args.n_rollout_threads - N_sol
    B_exp = args.B_exp
    del_switch = args.del_switch
    buffer_length = args.buffer_length
    h = args.h
    epsilon = args.epsilon
    delta = args.delta
    Rmin = args.Rmin
    Rmax = args.Rmax
    fixed_interval = args.fixed_interval
    historical_length = args.historical_length

    # mean_cover_rate = 0
    # starts = []
    random.seed(args.seed)
    np.random.seed(args.seed)
    node = node_buffer (args=args,
                        num_agents=num_agents,
                        buffer_length=buffer_length,
                        archive_initial_length=args.n_rollout_threads,
                        reproduction_num=B_exp,
                        epsilon=epsilon,
                        delta=delta)
    
    # run
    begin = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads // fixed_interval
    current_timestep = 0
    active_batch = args.n_rollout_threads
    all_batch = args.n_rollout_threads
    train_infos = {}

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           

        # reproduction
        node.archive += node.Sample_gradient(node.parent, current_timestep, h=h, use_gradient_noise=True)
        
        # reset env 
        starts, active_batch, all_batch = node.sample_starts(N_archive,N_sol)
        node.eval_score = np.zeros(shape=active_batch)

        for times in range(fixed_interval):
            obs = envs.new_starts_obs(starts, node.num_agents, all_batch)
            #replay buffer
            rollouts = RolloutStorage(num_agents,
                        args.episode_length, 
                        all_batch,
                        envs.observation_space[0], 
                        envs.action_space[0],
                        args.hidden_size) 
            # replay buffer init
            if args.share_policy: 
                share_obs = obs.reshape(all_batch, -1)        
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
            step_cover_rate = np.zeros(shape=(active_batch,args.episode_length))
            step_success = np.zeros(shape=(active_batch,args.episode_length))
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
                for i in range(all_batch):
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
                obs, rewards, dones, infos, _ = envs.step(actions_env, all_batch, num_agents)
                cover_rate_list = []
                success_list = []
                for env_id in range(active_batch):
                    cover_rate_list.append(infos[env_id][0]['cover_rate'])
                    success_list.append(int(infos[env_id][0]['success']))
                step_cover_rate[:,step] = np.array(cover_rate_list)
                step_success[:,step] = np.array(success_list)

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
                    share_obs = obs.reshape(all_batch, -1)        
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
            train_infos['training_cover_rate'] = np.mean(np.mean(step_cover_rate[:,-historical_length:],axis=1))
            current_timestep += args.episode_length * all_batch
            node.eval_score += np.mean(step_cover_rate[:,-historical_length:],axis=1)
                
            with torch.no_grad():  # get value and compute return
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
                value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(node.num_agents, rollouts, current_timestep,False) 
                rew = []
                for i in range(rollouts.rewards.shape[1]):
                    rew.append(np.sum(rollouts.rewards[:,i]))
                train_infos['value_loss'] = value_loss
                train_infos['average_episode_reward'] = np.mean(rew)
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
                    train_infos['average_episode_reward'] = np.mean(rew)
                    
                    rollouts[agent_id].after_update()

        # move nodes
        node.eval_score = node.eval_score / fixed_interval
        archive_length, parent_length, drop_num= node.update_buffer(active_batch, Rmax, Rmin, del_switch, current_timestep)
        train_infos['archive_length'] = archive_length
        train_infos['parent_length'] = parent_length
        train_infos['drop_num'] = drop_num
        if (episode+1) % args.save_node_interval ==0 and args.save_node:
            node.save_node(save_node_dir, episode)

        # test
        if episode % args.eval_interval==0:
            obs, _ = envs.reset(num_agents)
            episode_length = args.episode_length
            #replay buffer
            rollouts = RolloutStorage(num_agents,
                        episode_length, 
                        args.n_rollout_threads,
                        envs.observation_space[0], 
                        envs.action_space[0],
                        args.hidden_size) 
            # replay buffer init
            if args.share_policy: 
                share_obs = obs.reshape(args.n_rollout_threads, -1)        
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
            test_cover_rate = np.zeros(shape=(args.n_rollout_threads,episode_length))
            test_success = np.zeros(shape=(args.n_rollout_threads,episode_length))
            for step in range(episode_length):
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
                cover_rate_list = []
                success_list = []
                for env_id in range(args.n_rollout_threads):
                    cover_rate_list.append(infos[env_id][0]['cover_rate'])
                    success_list.append(int(infos[env_id][0]['success']))
                test_cover_rate[:,step] = np.array(cover_rate_list)
                test_success[:,step] = np.array(success_list)
                # test_cover_rate[:,step] = np.array(infos)[:,0]

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
            rew = []
            for i in range(rollouts.rewards.shape[1]):
                rew.append(np.sum(rollouts.rewards[:,i]))
            train_infos['eval_episode_reward'] = np.mean(rew)
            train_infos['eval_cover_rate'] = np.mean(np.mean(test_cover_rate[:,-historical_length:],axis=1))
            mean_cover_rate = np.mean(np.mean(test_cover_rate[:,-historical_length:],axis=1))
            # if mean_cover_rate >= 0.9 and args.algorithm_name=='ours' and save_90_flag:
            #     torch.save({'model': actor_critic}, str(save_dir) + "/cover09_agent_model.pt")
            #     save_90_flag = False

        total_num_steps = current_timestep

        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'state_dict': actor_critic.state_dict()
                            }, 
                            str(save_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'state_dict': actor_critic[agent_id].state_dict()
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
