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

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import shutil
import numpy as np
import imageio
import random
import pdb

def produce_good_case(num_case, start_boundary, now_agent_num, now_box_num):
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_box = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_box_num):
            landmark_location = np.random.uniform(-start_boundary, +start_boundary, 2)  
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        indices = random.sample(range(now_box_num), now_box_num)
        for k in indices:
            epsilon = -2 * 0.01 * random.random() + 0.01
            one_starts_box.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
        indices = random.sample(range(now_box_num), now_agent_num)
        for k in indices:
            epsilon = -2 * 0.01 * random.random() + 0.01
            one_starts_agent.append(copy.deepcopy(one_starts_box[k]+epsilon))
            # if epsilon < 0:
            #     one_starts_agent.append(copy.deepcopy(one_starts_box[k]+epsilon-0.25))
            # else:
            #     one_starts_agent.append(copy.deepcopy(one_starts_box[k]+epsilon+0.25))
        archive.append(one_starts_agent+one_starts_box+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
        one_starts_box = []
    return archive

def produce_good_case_grid(num_case, start_boundary, now_agent_num):
    # agent_size=0.1
    cell_size = 0.2
    grid_num = int(start_boundary * 2 / cell_size)
    grid = np.zeros(shape=(grid_num,grid_num))
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_agent_grid = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_agent_num):
            while 1:
                agent_location_grid = np.random.randint(0, grid.shape[0], 2) 
                if grid[agent_location_grid[0],agent_location_grid[1]]==1:
                    continue
                else:
                    grid[agent_location_grid[0],agent_location_grid[1]] = 1
                    one_starts_agent_grid.append(copy.deepcopy(agent_location_grid))
                    agent_location = np.array([(agent_location_grid[0]+0.5)*cell_size,(agent_location_grid[1]+0.5)*cell_size])-start_boundary
                    one_starts_agent.append(copy.deepcopy(agent_location))
                    break
        indices = random.sample(range(now_agent_num), now_agent_num)
        for k in indices:
            epsilons = np.array([[-1,0],[1,0],[0,1],[0,1],[1,1],[1,-1],[-1,1],[-1,-1]])
            epsilon = epsilons[random.sample(range(8),8)]
            for epsilon_id in range(epsilon.shape[0]):
                landmark_location_grid = one_starts_agent_grid[k] + epsilon[epsilon_id]
                if landmark_location_grid[0] > grid.shape[0]-1 or landmark_location_grid[1] > grid.shape[1]-1 \
                    or landmark_location_grid[0] <0 or landmark_location_grid[1] < 0:
                    continue
                if grid[landmark_location_grid[0],landmark_location_grid[1]]!=2:
                    grid[landmark_location_grid[0],landmark_location_grid[1]]=2
                    break
            landmark_location = np.array([(landmark_location_grid[0]+0.5)*cell_size,(landmark_location_grid[1]+0.5)*cell_size])-start_boundary
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        # select_starts.append(one_starts_agent+one_starts_landmark)
        archive.append(one_starts_agent+one_starts_landmark)
        grid = np.zeros(shape=(grid_num,grid_num))
        one_starts_agent = []
        one_starts_agent_grid = []
        one_starts_landmark = []
    return archive

def produce_hard_case(num_case, boundary, now_agent_num):
    one_starts_landmark = []
    one_starts_agent = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_agent_num-1):
            landmark_location = np.random.uniform(-boundary, -0.2*boundary, 2)  
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        landmark_location = np.random.uniform(0.8*boundary,boundary, 2)
        one_starts_landmark.append(copy.deepcopy(landmark_location))
        for i in range(now_agent_num):
            agent_location = np.random.uniform(-boundary, -0.5*boundary, 2)
            one_starts_agent.append(copy.deepcopy(agent_location))
        archive.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
    return archive

def produce_good_case_grid_pb(num_case, start_boundary, now_agent_num, now_box_num):
    # agent_size=0.2, ball_size=0.3,landmark_size=0.3
    cell_size = 0.3
    grid_num = int(start_boundary * 2 / cell_size)
    grid = np.zeros(shape=(grid_num,grid_num))
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_box = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_agent_num):
            while 1:
                agent_location_grid = np.random.randint(0, grid.shape[0], 2) 
                if grid[agent_location_grid[0],agent_location_grid[1]]==1:
                    continue
                else:
                    grid[agent_location_grid[0],agent_location_grid[1]] = 1
                    agent_location = np.array([(agent_location_grid[0]+0.5)*cell_size,(agent_location_grid[1]+0.5)*cell_size])-start_boundary
                    one_starts_agent.append(copy.deepcopy(agent_location))
                    break
        for i in range(now_box_num):
            while 1:
                box_location_grid = np.random.randint(0, grid.shape[0], 2) 
                if grid[box_location_grid[0],box_location_grid[1]]==1:
                    continue
                else:
                    grid[box_location_grid[0],box_location_grid[1]] = 1
                    box_location = np.array([(box_location_grid[0]+0.5)*cell_size,(box_location_grid[1]+0.5)*cell_size])-start_boundary
                    one_starts_box.append(copy.deepcopy(box_location))
                    break
        indices = random.sample(range(now_box_num), now_box_num)
        for k in indices:
            epsilons = np.array([[-0.3,0],[0.3,0],[0,-0.3],[0,0.3],[0.3,0.3],[0.3,-0.3],[-0.3,0.3],[-0.3,-0.3]])
            epsilon = epsilons[np.random.randint(0,8)]
            noise = -2 * 0.01 * random.random() + 0.01
            one_starts_landmark.append(copy.deepcopy(one_starts_box[k]+epsilon+noise))
        # select_starts.append(one_starts_agent+one_starts_landmark)
        archive.append(one_starts_agent+one_starts_box+one_starts_landmark)
        grid = np.zeros(shape=(grid_num,grid_num))
        one_starts_agent = []
        one_starts_landmark = []
        one_starts_box = []
    return archive

def produce_uniform_case(num_case, boundary, now_agent_num):
    one_starts_landmark = []
    one_starts_agent = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_agent_num):
            landmark_location = np.random.uniform(-boundary, +boundary, 2)  
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        for i in range(now_agent_num):
            agent_location = np.random.uniform(-boundary, +boundary, 2)
            one_starts_agent.append(copy.deepcopy(agent_location))
        archive.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
    return archive


def main():
    args = get_config()
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    run_dir = Path(args.model_dir)/ ("run" + str(args.seed)) / 'eval'
    if os.path.exists(run_dir): 
        shutil.rmtree(run_dir)
        os.mkdir(run_dir)
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    logger = SummaryWriter(str(log_dir))
    gifs_dir = run_dir / 'gifs'	
    os.makedirs(str(gifs_dir))
    
    num_agents = args.num_agents
    #Policy network 
    # if args.share_policy:   	       
    #     actor_critic = torch.load(str(args.model_dir) + 'run' + str(args.seed) + "/models/agent_model.pt")['model'].to(device)
    #     actor_critic.agents_num = num_agents
    # else:
    #     actor_critic = []
    #     for agent_id in range(num_agents):
    #         ac = torch.load(str(args.model_dir) + 'run' + str(args.seed) + "/models/agent" + str(agent_id) + "_model.pt")['model'].to(device)
    #         actor_critic.append(ac)
   
    actor_critic = torch.load('/home/tsing73/curriculum/results/MPE/simple_spread/optimizer2_true/run1/models/agent_model.pt')['model'].to(device)
    # actor_critic = torch.load('/home/chenjy/mappo-sc/results/MPE/push_ball/stage95_shaped_reward' + '/run2' + "/models/agent_model.pt")['model'].to(device)
    actor_critic.agents_num = 16
    actor_critic.boxes_num = 16
    num_agents = 16
    num_boxes = 16
    all_frames = []
    cover_rate = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    # starts = produce_good_case(num_case=1, start_boundary=0.3, now_agent_num=2, now_box_num=2)
    # archive 159
    # model_dir = '/home/chenjy/mappo-sc/node/MPE/simple_spread/stage95_warmup_3M_mean_8agents_big_batch/run1'
    # dir_path = model_dir  + '/8agents' + '/archive' + ('/archive_' + str(54))
    # if os.path.exists(dir_path):
    #     with open(dir_path,'r') as fp :
    #         data = fp.readlines()
    #     for i in range(len(data)):
    #         if len(data[i])<100:
    #             data[i-1] = data[i-1][:-1] + data[i]
    #     for i in range(len(data)):
    #         data[i] = np.array(data[i][1:-2].split(),dtype=float)
    #     data_true = []
    #     for i in range(len(data)):
    #         if data[i].shape[0]>5:
    #             data_true.append(data[i])
    # starts = data_true
    # starts = produce_good_case_grid_pb(500,0.3,num_agents,num_boxes)
    starts = produce_good_case_grid(500,3.0,num_agents)
    # starts = produce_hard_case(500,3,8)
    # starts = produce_uniform_case(500,3,16)
    for eval_episode in range(args.eval_episodes):
        print(eval_episode)
        eval_env = MPEEnv(args)
        if args.save_gifs:
            image = eval_env.render('rgb_array', close=False)[0]
            all_frames.append(image)
        
        # eval_obs, _ = eval_env.reset(num_agents,num_boxes)
        # eval_obs, _ = eval_env.reset(num_agents)
        eval_obs = eval_env.new_starts_obs(starts[eval_episode],num_agents)
        # eval_obs = eval_env.new_starts_obs_pb(starts[eval_episode],num_agents,num_boxes)
        eval_obs = np.array(eval_obs)       
        eval_share_obs = eval_obs.reshape(1, -1)
        eval_recurrent_hidden_states = np.zeros((num_agents,args.hidden_size)).astype(np.float32)
        eval_recurrent_hidden_states_critic = np.zeros((num_agents,args.hidden_size)).astype(np.float32)
        eval_masks = np.ones((num_agents,1)).astype(np.float32)
        
        for step in range(args.episode_length): 
            calc_start = time.time()              
            eval_actions = []            
            for agent_id in range(num_agents):
                if args.share_policy:
                    actor_critic.eval()
                    _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                        torch.FloatTensor(eval_share_obs), 
                        torch.FloatTensor(eval_obs[agent_id].reshape(1,-1)), 
                        torch.FloatTensor(eval_recurrent_hidden_states[agent_id]), 
                        torch.FloatTensor(eval_recurrent_hidden_states_critic[agent_id]),
                        torch.FloatTensor(eval_masks[agent_id]),
                        None,
                        deterministic=True)
                else:
                    actor_critic[agent_id].eval()
                    _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                        torch.FloatTensor(eval_share_obs), 
                        torch.FloatTensor(eval_obs[agent_id]), 
                        torch.FloatTensor(eval_recurrent_hidden_states[agent_id]), 
                        torch.FloatTensor(eval_recurrent_hidden_states_critic[agent_id]),
                        torch.FloatTensor(eval_masks[agent_id]),
                        None,
                        deterministic=True)
    
                eval_actions.append(action.detach().cpu().numpy())
                eval_recurrent_hidden_states[agent_id] = recurrent_hidden_states.detach().cpu().numpy()
                eval_recurrent_hidden_states_critic[agent_id] = recurrent_hidden_states_critic.detach().cpu().numpy()
    
            # rearrange action           
            eval_actions_env = []
            for agent_id in range(num_agents):
                one_hot_action = np.zeros(eval_env.action_space[0].n)
                one_hot_action[eval_actions[agent_id][0]] = 1
                eval_actions_env.append(one_hot_action)
                    
            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos, _ = eval_env.step(eval_actions_env)
            # print('reward: ', eval_rewards)
            eval_obs = np.array(eval_obs)
            eval_share_obs = eval_obs.reshape(1, -1)
            
            if args.save_gifs:
                image = eval_env.render('rgb_array', close=False)[0]
                all_frames.append(image)
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < args.ifi:
                    time.sleep(ifi - elapsed)
        print('cover_rate: ', np.mean(eval_infos))
        cover_rate += np.mean(eval_infos)                      
        if args.save_gifs:
            gif_num = 0
            imageio.mimsave(str(gifs_dir / args.scenario_name) + '_%i.gif' % gif_num,
                        all_frames, duration=args.ifi)  
        # if save_gifs:
        #     gif_num = 0
        #     while os.path.exists('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i)):
        #         gif_num += 1
        #     imageio.mimsave('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i),
        #                     frames, duration=ifi)
    print('average_cover_rate: ', cover_rate/args.eval_episodes)        
    eval_env.close()
if __name__ == "__main__":
    main()
