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
from algorithm.model import Policy, Policy3, ATTBase_actor_sl, ATTBase_critic_sl

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
# wandb.init(project="my-project")
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

class node_buffer():
    def __init__(self,agent_num,landmark_num,buffer_length,archive_initial_length,reproduction_num,max_step,start_boundary,boundary,legal_region,epsilon,delta):
        self.agent_num = agent_num
        self.landmark_num = landmark_num
        self.buffer_length = buffer_length
        self.archive_initial_length = archive_initial_length
        self.archive = []
        self.initial_archive = self.produce_good_case_sl(archive_initial_length, start_boundary, self.agent_num, self.landmark_num)
        self.archive_score = np.zeros(len(self.initial_archive))
        # self.archive_novelty = self.get_novelty(self.initial_archive,self.initial_archive)
        # self.initial_archive, self.archive_novelty = self.novelty_sort(self.initial_archive, self.archive_novelty)
        # self.archive, self.archive_novelty, self.archive_score = self.novelty_score_sort(self.archive, self.archive_novelty, self.archive_score)
        self.childlist = []
        self.hardlist = []
        self.parent = []
        self.parent_all = []
        self.max_step = max_step
        self.boundary = boundary
        self.legal_region = legal_region
        self.reproduction_num = reproduction_num
        self.choose_child_index = []
        self.choose_archive_index = []
        self.eval_score = np.zeros(shape=len(self.initial_archive))
        self.topk = 5
        self.epsilon = epsilon
        self.delta = delta

    def produce_good_case(self, num_case, start_boundary, now_agent_num):
        one_starts_landmark = []
        one_starts_agent = []
        archive = [] 
        for j in range(num_case):
            for i in range(now_agent_num):
                # landmark_location = np.random.uniform(-start_boundary, +start_boundary, 2) 
                landmark_location = np.array([np.random.uniform(start_boundary[0],start_boundary[1]),np.random.uniform(start_boundary[2],start_boundary[3])])
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

    def produce_good_case_sl(self, num_case, start_boundary, now_agent_num, now_landmark_num):
        one_starts_landmark = []
        one_starts_agent = []
        archive = [] 
        for j in range(num_case):
            for i in range(now_landmark_num):
                # landmark_location = np.random.uniform(-start_boundary, +start_boundary, 2) 
                landmark_location = np.array([np.random.uniform(start_boundary[0],start_boundary[1]),np.random.uniform(start_boundary[2],start_boundary[3])])
                one_starts_landmark.append(copy.deepcopy(landmark_location))
            # index_sample = BatchSampler(SubsetRandomSampler(range(now_agent_num)),now_agent_num,drop_last=True)
            indices = random.sample(range(now_landmark_num), now_agent_num)
            for k in indices:
                epsilon = -2 * 0.01 * random.random() + 0.01
                one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
            # select_starts.append(one_starts_agent+one_starts_landmark)
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
    
    def novelty_score_sort(self, buffer, buffer_novelty, buffer_score):
        zipped = zip(buffer,buffer_novelty,buffer_score)
        sort_zipped = sorted(zipped,key=lambda x:(x[1],np.mean(x[0])))
        result = zip(*sort_zipped)
        buffer_new, buffer_novelty_new, buffer_score_new = [list(x) for x in result]
        return buffer_new, buffer_novelty_new, buffer_score_new

    def SampleNearby_novelty(self, parents, child_novelty_threshold, writer, timestep): # produce high novelty children and return 
        if len(self.parent_all) > self.topk + 1:
            self.parent_all_novelty = self.get_novelty(self.parent_all,self.parent_all)
            self.parent_all, self.parent_all_novelty = self.novelty_sort(self.parent_all, self.parent_all_novelty)
            novelty_threshold = np.mean(self.parent_all_novelty)
        else:
            novelty_threshold = 0
        # novelty_threshold = child_novelty_threshold
        wandb.log({str(self.agent_num)+'novelty_threshold': novelty_threshold},timestep)
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

    def SampleNearby_novelty_activeAndsolve(self, parents, child_novelty_threshold, writer, timestep): # produce high novelty children and return 
        self.activeAndsolve = self.archive + self.parent_all
        if len(self.activeAndsolve) > self.topk + 1:
            self.activeAndsolve_novelty = self.get_novelty(self.activeAndsolve,self.activeAndsolve)
            self.activeAndsolve, self.activeAndsolve_novelty = self.novelty_sort(self.activeAndsolve, self.activeAndsolve_novelty)
            novelty_threshold = np.mean(self.activeAndsolve_novelty)
        else:
            novelty_threshold = 0
        # novelty_threshold = child_novelty_threshold
        wandb.log({str(self.agent_num)+'novelty_threshold': novelty_threshold},timestep)
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
                    if len(self.activeAndsolve) > self.topk + 1:
                        if self.get_novelty([st],self.activeAndsolve) > novelty_threshold:
                            child_new.append(copy.deepcopy(st))
                            add_num += 1
                    else:
                        child_new.append(copy.deepcopy(st))
                        add_num += 1
            child_new = random.sample(child_new, min(self.reproduction_num,len(child_new)))
            return child_new

    def Sample_gradient(self,parents,timestep,h=100, use_gradient_noise=True):
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
                        if parent_of_entity_id < self.agent_num:
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
        print('sample_archive: ', len(self.choose_archive_index))
        print('sample_childlist: ', len(self.choose_child_index))
        print('sample_parent: ', len(self.choose_parent_index))
        return starts, one_length, starts_length

    def sample_starts_wo_evaluation(self, N_archive, N_parent):
        # sample uniformly from archive and initial_archive
        # check archive (>Rmax and < Rmin drop), particles from initial_archive (only drop > Rmax)
        
        self.choose_parent_index = random.sample(range(len(self.parent_all)),min(len(self.parent_all), N_parent))
        N_archive = N_archive + N_parent - len(self.choose_parent_index)
        if len(self.initial_archive) > 0:
            if len(self.archive) > 0:
                N_initial_archive = int(N_archive * len(self.initial_archive) / (len(self.archive) + len(self.initial_archive)))
            else:
                N_initial_archive = N_archive
        else:
            N_initial_archive = 0
        
        N_archive = N_archive - N_initial_archive

        self.choose_initial_archive_index = random.sample(range(len(self.initial_archive)), min(len(self.initial_archive), N_initial_archive))
        self.choose_archive_index = random.sample(range(len(self.archive)),min(len(self.archive), N_archive + N_initial_archive + N_parent - len(self.choose_parent_index) - len(self.choose_initial_archive_index)))
        if len(self.choose_initial_archive_index) + len(self.choose_archive_index) < N_archive + N_initial_archive:
            self.choose_parent_index = random.sample(range(len(self.parent_all)), min(len(self.parent_all), N_archive + N_initial_archive + N_parent - len(self.choose_archive_index) - len(self.choose_initial_archive_index)))
        
        self.choose_initial_archive_index = np.sort(self.choose_initial_archive_index)
        self.choose_archive_index = np.sort(self.choose_archive_index)
        self.choose_parent_index = np.sort(self.choose_parent_index)
        one_length = len(self.choose_initial_archive_index) + len(self.choose_archive_index)
        starts_length = len(self.choose_initial_archive_index) + len(self.choose_archive_index) + len(self.choose_parent_index)
        starts = []
        for i in range(len(self.choose_initial_archive_index)):
            starts.append(self.initial_archive[self.choose_initial_archive_index[i]])
        for i in range(len(self.choose_archive_index)):
            starts.append(self.archive[self.choose_archive_index[i]])
        for i in range(len(self.choose_parent_index)):
            starts.append(self.parent_all[self.choose_parent_index[i]])
        print('sample_initail_archive: ', len(self.choose_initial_archive_index))
        print('sample_archive: ', len(self.choose_archive_index))
        print('sample_parent: ', len(self.choose_parent_index))
        return starts, one_length, starts_length

    def move_nodes(self, one_length, Rmax, Rmin, use_child_novelty, use_parent_novelty, child_novelty_threshold, del_switch, writer, timestep): 
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
        if use_child_novelty and len(child2archive)!=0:
            child2archive_novelty = self.get_novelty(child2archive,self.parent_all)
            child2archive, child2archive_novelty = self.novelty_sort(child2archive,child2archive_novelty)
            for i in range(len(child2archive)):
                if child2archive_novelty[i] > child_novelty_threshold:
                    self.archive.append(child2archive[i])
            # child2archive = child2archive[int(len(child2archive)/2):]
        else:
            self.archive += child2archive
        # self.archive += child2archive
        if use_parent_novelty:
            start_sort = time.time()
            parent_novelty = []
            if len(self.parent_all) > self.topk+1 and self.parent!=[]:
                parent_novelty = self.get_novelty(self.parent,self.parent_all)
                self.parent, parent_novelty = self.novelty_sort(self.parent, parent_novelty)
                self.parent = self.parent[int(len(self.parent)/2):]
            end_sort = time.time()
            print('sort_archive: ', end_sort-start_sort)
        self.parent_all += self.parent
        print('child_drop: ', drop_num)
        print('add_hard_num: ', add_hard_num )
        print('parent: ', len(self.parent))
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
        wandb.log({str(self.agent_num)+'archive_length': len(self.archive)},timestep)
        wandb.log({str(self.agent_num)+'childlist_length': len(self.childlist)},timestep)
        wandb.log({str(self.agent_num)+'parentlist_length': len(self.parent)},timestep)
        wandb.log({str(self.agent_num)+'drop_num': drop_num},timestep)

    def move_nodes_Qact_Qsol(self, one_length, Rmax, Rmin, del_switch, timestep):
        del_archive_num = 0
        del_initial_archive_num = 0
        del_easy_num = 0
        add_hard_num = 0
        self.parent = []
        for i in range(one_length):
            if i >= len(self.choose_initial_archive_index): # deal with archive
                if self.eval_score[i] > Rmax:
                    self.parent.append(copy.deepcopy(self.archive[self.choose_archive_index[i - len(self.choose_initial_archive_index)]-del_archive_num]))
                    del self.archive[self.choose_archive_index[i - len(self.choose_initial_archive_index)]-del_archive_num]
                    del_archive_num += 1
                elif self.eval_score[i] < Rmin:
                    del self.archive[self.choose_archive_index[i - len(self.choose_initial_archive_index)]-del_archive_num]
                    del_archive_num += 1
            else: # deal with initial archive
                if self.eval_score[i] > Rmax:
                    self.parent.append(copy.deepcopy(self.initial_archive[self.choose_initial_archive_index[i]-del_initial_archive_num]))
                    del self.initial_archive[self.choose_initial_archive_index[i]-del_initial_archive_num]
                    del_initial_archive_num += 1
        self.parent_all += self.parent
        if len(self.archive) > self.buffer_length:
            if del_switch=='novelty' : # novelty del
                self.archive_novelty = self.get_novelty(self.archive,self.archive)
                self.archive,self.archive_novelty = self.novelty_sort(self.archive,self.archive_novelty)
                # self.archive, self.archive_novelty, self.archive_score = self.novelty_score_sort(self.archive, self.archive_novelty, self.archive_score)
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
        wandb.log({str(self.agent_num)+'initial_archive_length': len(self.initial_archive)},timestep)
        wandb.log({str(self.agent_num)+'archive_length': len(self.archive)},timestep)
        wandb.log({str(self.agent_num)+'parentlist_length': len(self.parent)},timestep)
        wandb.log({str(self.agent_num)+'del_num': del_archive_num},timestep)

    def save_node(self, dir_path, episode):
        # dir_path: '/home/chenjy/mappo-curriculum/' + args.model_dir
        if self.agent_num!=0:
            save_path = dir_path / ('%iagents' % (self.agent_num))
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
                self.archive_novelty = self.get_novelty(self.archive,self.archive)
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
    run = wandb.init(project='simple_speaker_listener',name=str(args.algorithm_name) + "_seed" + str(args.seed))
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
    num_landmarks = args.num_landmarks
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
    else:
        actor_critic = []
        agents = []
        rollouts = []
        obs_role_dim = {'speaker': envs.observation_space[0].shape[0], 'listener': envs.observation_space[1].shape[0]}
        actor_base = {'speaker': ATTBase_actor_sl(envs.observation_space[0].shape[0], envs.action_space[0], num_agents, role='speaker'),
                        'listener': ATTBase_actor_sl(envs.observation_space[1].shape[0], envs.action_space[1], num_agents, role='listener')}
        critic_base = ATTBase_critic_sl(num_agents, obs_role_dim)
        for agent_id in range(num_agents):
            if agent_id == 0:
                role = 'speaker'
            else:
                role = 'listener'
            ac = Policy3(envs.observation_space[0], 
                        envs.action_space[0],
                        num_agents = num_agents,
                        num_landmarks = num_landmarks,
                        base=None,
                        actor_base=actor_base[role],
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

            ac.to(device)
            # algorithm
            agent = PPO3(ac,
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
    
    use_parent_sample = True
    use_gradient_sample = True
    del_switch = 'novelty'
    starts = []
    buffer_length = 2000 # archive 长度
    N_parent = 5
    N_archive = args.n_rollout_threads - N_parent
    h = 1
    epsilon = 0.2
    delta = 0.2
    B_exp = 30 # equal to curriculum_sp
    Rmin = 0.5
    Rmax = 0.95
    boundary = 1
    start_boundary = [-0.3,0.3,-0.3,0.3] # 分别代表x的范围和y的范围
    legal_region = {'agent':{'x':[[-1,1]],'y': [[-1,1]]},
        'landmark':{'x':[[-1,1]],'y': [[-1,1]]}} # legal region for samplenearby
    max_step = 0.2
    N_easy = 0
    test_flag = 0
    reproduce_flag = 0
    target_num = args.num_agents
    last_agent_num = args.num_agents
    now_agent_num = args.num_agents
    mean_cover_rate = 0
    eval_frequency = 3 #需要fix几个回合
    check_frequency = 1
    save_node_frequency = 5
    save_node_flag = True
    save_90_flag = True
    historical_length = 5
    random.seed(args.seed)
    np.random.seed(args.seed)
    last_node = node_buffer(num_agents, num_landmarks, buffer_length,
                           archive_initial_length=args.n_rollout_threads,
                           reproduction_num=B_exp,
                           max_step=max_step,
                           start_boundary=start_boundary,
                           boundary=boundary,
                           legal_region=legal_region,
                           epsilon=epsilon,
                           delta=delta)
    
    # run
    begin = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads // eval_frequency
    curriculum_episode = 0
    current_timestep = 0
    one_length = args.n_rollout_threads
    starts_length = args.n_rollout_threads

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           

        # reproduction
        if use_gradient_sample:
            # last_node.archive += last_node.Sample_gradient(last_node.parent, current_timestep,h=h, use_gradient_noise=True)
            last_node.archive += last_node.Sample_gradient(last_node.parent, current_timestep,h=h, use_gradient_noise=True)
        
        # reset env 
        if use_parent_sample:
            starts, one_length, starts_length = last_node.sample_starts_wo_evaluation(N_archive,N_parent)
        else:
            starts, one_length, starts_length = last_node.sample_starts(N_child,N_archive)
        last_node.eval_score = np.zeros(shape=one_length)

        for times in range(eval_frequency):
            obs = envs.new_starts_obs_sl(starts, starts_length)

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
            step_success = np.zeros(shape=(one_length,args.episode_length))
            for step in range(args.episode_length):
                # Sample actions
                values = []
                actions= []
                action_log_probs = []
                recurrent_hidden_statess = []
                recurrent_hidden_statess_critic = []
                
                with torch.no_grad():                
                    for agent_id in range(num_agents):
                        role_id = 'speaker' if agent_id == 0 else 'listener'
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
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act_role(agent_id,
                                torch.FloatTensor(rollouts[agent_id].share_obs[step,:]), 
                                torch.FloatTensor(rollouts[agent_id].obs[step,:]), 
                                role_id,
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
            current_timestep += args.episode_length * starts_length
            curriculum_episode += 1
            last_node.eval_score += np.mean(step_cover_rate[:,-historical_length:],axis=1)
                
            with torch.no_grad():  # get value and compute return
                for agent_id in range(num_agents):    
                    role_id = 'speaker' if agent_id == 0 else 'listener'   
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
                        next_value,_,_ = actor_critic[agent_id].get_value_role(agent_id,
                                                                torch.FloatTensor(rollouts[agent_id].share_obs[-1,:]), 
                                                                torch.FloatTensor(rollouts[agent_id].obs[-1,:]), 
                                                                role_id,
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
                value_loss, action_loss, dist_entropy = agents.update_share_asynchronous(last_node.agent_num, rollouts, current_timestep,False) 
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
                    role_id = 'speaker' if agent_id == 0 else 'listener'
                    actor_critic[agent_id].train()
                    value_loss, action_loss, dist_entropy = agents[agent_id].update_single(agent_id, role_id, rollouts[agent_id],current_timestep)
                    value_losses.append(value_loss)
                    action_losses.append(action_loss)
                    dist_entropies.append(dist_entropy)

                    rew = []
                    for i in range(rollouts[agent_id].rewards.shape[1]):
                        rew.append(np.sum(rollouts[agent_id].rewards[:,i]))
                    wandb.log({'value_loss_%i'%agent_id: value_loss},current_timestep)
                    rew = []
                    for i in range(rollouts[agent_id].rewards.shape[1]):
                        rew.append(np.sum(rollouts[agent_id].rewards[:,i]))
                    wandb.log({'training_average_episode_reward_%i'%agent_id: np.mean(rew)},current_timestep)
                    
                    rollouts[agent_id].after_update()

        # move nodes
        last_node.eval_score = last_node.eval_score / eval_frequency
        last_node.move_nodes_Qact_Qsol(one_length, Rmax, Rmin, del_switch, current_timestep)
        print('last_node_parent: ', len(last_node.parent))
        if (episode+1) % save_node_frequency ==0 and save_node_flag:
            last_node.save_node(save_node_dir, episode)
        print('archive: ', len(last_node.archive))

        # test
        if episode % check_frequency==0:
            obs, _ = envs.reset(num_agents)
            episode_length = args.episode_length
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
            for step in range(episode_length):
                # Sample actions
                values = []
                actions= []
                action_log_probs = []
                recurrent_hidden_statess = []
                recurrent_hidden_statess_critic = []
                
                with torch.no_grad():                
                    for agent_id in range(num_agents):
                        role_id = 'speaker' if agent_id == 0 else 'listener'
                        if args.share_policy:
                            actor_critic.eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act_role(agent_id,
                                torch.FloatTensor(rollouts.share_obs[step,:,agent_id]), 
                                torch.FloatTensor(rollouts.obs[step,:,agent_id]), 
                                torch.FloatTensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                                torch.FloatTensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                                torch.FloatTensor(rollouts.masks[step,:,agent_id]),deterministic=True)
                        else:
                            actor_critic[agent_id].eval()
                            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act_role(agent_id,
                                torch.FloatTensor(rollouts[agent_id].share_obs[step,:]), 
                                torch.FloatTensor(rollouts[agent_id].obs[step,:]), 
                                role_id,
                                torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states[step,:]), 
                                torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states_critic[step,:]),
                                torch.FloatTensor(rollouts[agent_id].masks[step,:]),
                                deterministic=True)
                            
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
                for env_id in range(args.n_rollout_threads):
                    cover_rate_list.append(infos[env_id][0]['cover_rate'])
                test_cover_rate[:,step] = np.array(cover_rate_list)

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
            for i in range(rollouts[0].rewards.shape[1]):
                rew.append(np.sum(rollouts[0].rewards[:,i]))
            wandb.log(
                {'eval_episode_reward': np.mean(rew)},
                current_timestep)
            wandb.log({'cover_rate_1step': np.mean(test_cover_rate[:,-1])},current_timestep)
            wandb.log({'cover_rate_5step': np.mean(np.mean(test_cover_rate[:,-historical_length:],axis=1))}, current_timestep)
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