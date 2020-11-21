import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import imageio
#from a2c_ppo_acktr.model import MLPBase, CNNBase
import pdb
import time
import copy
import random
from sklearn.decomposition import PCA
from pathlib import Path
from config import get_config

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

def produce_good_case_sp(num_case, start_boundary, now_agent_num):
    one_starts_landmark = []
    one_starts_agent = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_agent_num):
            landmark_location = np.random.uniform(-start_boundary, +start_boundary, 2) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        index_sample = BatchSampler(SubsetRandomSampler(range(now_agent_num)),now_agent_num,drop_last=True)
        for indices in index_sample:
            for k in indices:
                one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+0.01))
        # pdb.set_trace()
        # select_starts.append(one_starts_agent+one_starts_landmark)
        archive.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
    return archive

def produce_good_case_pb(num_case, start_boundary, now_agent_num, now_box_num):
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_box = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_box_num):
            landmark_location = np.random.uniform(-start_boundary, +start_boundary, 2)  
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        # index_sample = BatchSampler(SubsetRandomSampler(range(now_agent_num)),now_agent_num,drop_last=True)
        indices = random.sample(range(now_agent_num), now_agent_num)
        for k in indices:
            epsilon = -2 * 0.01 * random.random() + 0.01
            one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
        indices = random.sample(range(now_box_num), now_box_num)
        for k in indices:
            epsilon = -2 * 0.01 * random.random() + 0.01
            one_starts_box.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
        # select_starts.append(one_starts_agent+one_starts_landmark)
        archive.append(one_starts_agent+one_starts_box+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
        one_starts_box = []
    return archive

def uniform_case_pb(num_case, start_boundary, now_agent_num, now_box_num):
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_box = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_box_num):
            landmark_location = np.random.uniform(-start_boundary, +start_boundary, 2)  
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        for i in range(now_agent_num):
            agent_location = np.random.uniform(-start_boundary, +start_boundary, 2)  
            one_starts_agent.append(copy.deepcopy(agent_location))
        for i in range(now_box_num):
            box_location = np.random.uniform(-start_boundary, +start_boundary, 2)  
            one_starts_box.append(copy.deepcopy(box_location))

        # select_starts.append(one_starts_agent+one_starts_landmark)
        archive.append(one_starts_agent+one_starts_box+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
        one_starts_box = []
    return archive

def produce_uniform_case_H(num_case, now_agent_num): # 产生H_map的随机态
    one_starts_landmark = []
    one_starts_agent = []
    archive = [] 
    # boundary_x x轴活动范围
    # boundary_y 是y轴活动范围
    boundary_x = [[-4.9,-3.1],[-3,-1],[-0.9,0.9],[1,3],[3.1,4.9]]
    boundary_y = [[-2.9,2.9],[-0.15,0.15],[-2.9,2.9],[-0.15,0.15],[-2.9,2.9]]
    for j in range(num_case):
        for i in range(now_agent_num):
            location_id = np.random.choice([0,1,2,3,4],1)[0]
            landmark_location_x = np.random.uniform(boundary_x[location_id][0],boundary_x[location_id][1])
            landmark_location_y = np.random.uniform(boundary_y[location_id][0],boundary_y[location_id][1])
            landmark_location = np.array([landmark_location_x,landmark_location_y])
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        for i in range(now_agent_num):
            location_id = np.random.choice([0,1,2,3,4],1)[0]
            agent_location_x = np.random.uniform(boundary_x[location_id][0],boundary_x[location_id][1])
            agent_location_y = np.random.uniform(boundary_y[location_id][0],boundary_y[location_id][1])
            agent_location = np.array([agent_location_x,agent_location_y])
            one_starts_landmark.append(copy.deepcopy(agent_location))
        archive.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
    return archive

# Parameters
gamma = 0.95
render = False
seed = 0
log_interval = 10
def num_reach(world):
    num = 0
    for l in world.landmarks:
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        if min(dists) <= world.agents[0].size + world.landmarks[0].size:
            num = num + 1
    return num 

if __name__ == '__main__':
    args = get_config()
    # env = make_env(args.env_name, discrete_action=True)
    num_processes = 1000
    # envs = make_parallel_env(args.env_name, num_processes, args.seed, True)
    cover_rate_sum = 0
    agent_num = 4
    box_num = 4
    data = []
    # # uniform data
    # starts = produce_uniform_case_H(10000, 4)
    # with open('/home/tsing73/curriculum/node_data/sp3_10*6.txt','w') as fp:
    #     for i in range(len(starts)):
    #         fp.write(str(np.array(starts[i]).reshape(-1))+'\n')

    mode_path = Path('./node') / args.env_name / args.scenario_name / args.algorithm_name / 'run1'
    if args.scenario_name=='simple_spread' or args.scenario_name=='simple_spread_H':
        mode_path = mode_path / ('%iagents'%agent_num)
        data_dir = '/home/tsing73/curriculum/node_data/sp_66_%i.txt' %agent_num
    elif args.scenario_name=='push_ball':
        mode_path = mode_path / ('%iagents'%box_num)
        data_dir = '/home/tsing73/curriculum/node_data/pb_44_2people%ibox.txt' %box_num
    elif args.scenario_name=='simple_spread_3rooms':
        mode_path = mode_path / ('%iagents'%agent_num)
        data_dir = '/home/tsing73/curriculum/node_data/sp3_10*6.txt'
    with open(data_dir,'r') as fp:
        data = fp.readlines()
    for i in range(len(data)):
        if len(data[i])<100:
            data[i-1] = data[i-1][:-1] + data[i]
    for i in range(len(data)):
        data[i] = np.array(data[i][1:-2].split(),dtype=float)
    data_true = []
    for i in range(len(data)):
        if data[i].shape[0]>5:
            data_true.append(data[i])
    data = np.array(data_true)
    pca = PCA(n_components=2)
    pca.fit(data)
    # load from files
    archive = []
    archive_novelty = []
    j = 0
    while j<=1000:
        dir_path = mode_path  / 'archive' / ('archive_' + str(j))
        dir_path2 = mode_path  / 'archive_novelty' / ('archive_novelty' + str(j))
        if os.path.exists(dir_path):
            with open(dir_path,'r') as fp :
                archive = fp.readlines()

            # for i in range(len(archive)):
            #     if len(archive[i])<100:
            #         archive[i-1] = archive[i-1][:-1] + archive[i]
            for i in range(len(archive)):
                archive[i] = np.array(archive[i][1:-2].split(),dtype=float)
            archive_true = []
            for i in range(len(archive)):
                if archive[i].shape[0]>5:
                    archive_true.append(archive[i])

            # for i in range(len(archive)):
            #     archive[i] = np.array(archive[i][1:-2].split(),dtype=float)
            # with open(dir_path2,'r') as fp :
            #     archive_novelty = fp.readlines()
            # for i in range(len(archive_novelty)):
            #     archive_novelty[i] = np.array(archive_novelty[i][1:-2].split(),dtype=float)
            archive = np.array(archive_true)
            archive_novelty = []
            # archive_novelty = np.array(archive_novelty)[len(archive_novelty)-len(archive):]
            uniform = pca.transform(data)
            if archive.shape[0]!=0:
                archive_projection = pca.transform(archive)
                plt.cla()
                plt.scatter(uniform[:, 0], uniform[:, 1], marker='o')
                archive_novelty = []
                if len(archive_novelty)==0:
                    plt.scatter(archive_projection[:, 0], archive_projection[:, 1], marker='1')
                else:
                    plt.scatter(archive_projection[:, 0], archive_projection[:, 1], marker='1', c=archive_novelty)
                plt.savefig(mode_path / 'archive' / ('result_%i.jpg'%j))
        # dir_path = mode_path  / 'childlist' / ('child_' + str(j))
        # if os.path.exists(dir_path):
        #     with open(dir_path,'r') as fp :
        #         archive = fp.readlines()
        #     for i in range(len(archive)):
        #         archive[i] = np.array(archive[i][1:-2].split(),dtype=float)
        #     archive = np.array(archive)
        #     uniform = pca.transform(data)
        #     if archive.shape[0]!=0:
        #         archive_projection = pca.transform(archive)
        #         plt.cla()
        #         plt.scatter(uniform[:, 0], uniform[:, 1], marker='o')
        #         plt.scatter(archive_projection[:, 0], archive_projection[:, 1],marker='1')
        #         plt.savefig(mode_path / 'childlist' / ('result_%i.jpg'%j))
        # dir_path = mode_path  / 'parent_all' / ('parent_all_' + str(j))
        # if os.path.exists(dir_path):
        #     with open(dir_path,'r') as fp :
        #         archive = fp.readlines()
        #     for i in range(len(archive)):
        #         archive[i] = np.array(archive[i][1:-2].split(),dtype=float)
        #     archive = np.array(archive)
        #     uniform = pca.transform(data)
        #     if archive.shape[0]!=0:
        #         archive_projection = pca.transform(archive)
        #         plt.cla()
        #         plt.scatter(uniform[:, 0], uniform[:, 1], marker='o')
        #         plt.scatter(archive_projection[:, 0], archive_projection[:, 1],marker='1')
        #         plt.savefig(mode_path / 'parent_all' / ('result_%i.jpg'%j))
        j += 1


