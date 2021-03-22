#!/usr/bin/env python

import copy
import os
import time
import numpy as np
from pathlib import Path

import torch
from tensorboardX import SummaryWriter

from envs import MPEEnv

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import shutil
import numpy as np
import imageio
import random
import pdb
np.set_printoptions(linewidth=26)

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
    cell_size = 0.3
    grid_num = round((start_boundary[1]-start_boundary[0]) / cell_size)
    init_origin_node = np.array([start_boundary[0]+0.5*cell_size,start_boundary[3]-0.5*cell_size]) # left, up
    assert grid_num ** 2 >= now_agent_num + now_box_num
    grid = np.zeros(shape=(grid_num,grid_num))
    grid_without_landmark = np.zeros(shape=(grid_num,grid_num))
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_box = []
    one_starts_box_grid = []
    archive = [] 
    for j in range(num_case):
        # box location
        for i in range(now_box_num):
            while 1:
                box_location_grid = np.random.randint(0, grid.shape[0], 2) 
                if grid[box_location_grid[0],box_location_grid[1]]==1:
                    continue
                else:
                    grid[box_location_grid[0],box_location_grid[1]] = 1
                    # box_location = np.array([(box_location_grid[0]+0.5)*cell_size,(box_location_grid[1]+0.5)*cell_size])-start_boundary
                    box_location = np.array([(box_location_grid[0]+0.5)*cell_size,-(box_location_grid[1]+0.5)*cell_size]) + init_origin_node
                    one_starts_box.append(copy.deepcopy(box_location))
                    one_starts_box_grid.append(copy.deepcopy(box_location_grid))
                    break
        grid_without_landmark = copy.deepcopy(grid)
        # landmark location
        indices = random.sample(range(now_box_num), now_box_num)
        num_try = 0
        num_tries = 20
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
                    landmark_location = np.array([(landmark_location_grid[0]+0.5)*cell_size,-(landmark_location_grid[1]+0.5)*cell_size]) + init_origin_node
                    one_starts_landmark.append(copy.deepcopy(landmark_location))
                    break
        # agent_location
        indices_agent = random.sample(range(now_box_num), now_box_num)
        num_try = 0
        num_tries = 20
        for k in indices_agent:
            around = 1
            while num_try < num_tries:
                delta_x_direction = random.randint(-around,around)
                delta_y_direction = random.randint(-around,around)
                agent_location_x = min(max(0,one_starts_box_grid[k][0]+delta_x_direction),grid.shape[0]-1)
                agent_location_y = min(max(0,one_starts_box_grid[k][1]+delta_y_direction),grid.shape[1]-1)
                agent_location_grid = np.array([agent_location_x,agent_location_y])
                if grid_without_landmark[agent_location_grid[0],agent_location_grid[1]]==1:
                    num_try += 1
                    if num_try >= num_tries and around==1:
                        around = 2
                        num_try = 0
                    assert num_try<num_tries or around==1, 'case %i can not find agent pos'%j
                    continue
                else:
                    grid_without_landmark[agent_location_grid[0],agent_location_grid[1]] = 1
                    # agent_location = np.array([(agent_location_grid[0]+0.5)*cell_size,(agent_location_grid[1]+0.5)*cell_size])-start_boundary
                    agent_location = np.array([(agent_location_grid[0]+0.5)*cell_size,-(agent_location_grid[1]+0.5)*cell_size]) + init_origin_node
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

def produce_uniform_case_grid(num_case, start_boundary, now_agent_num):
    # agent_size=0.1
    cell_size = 0.2
    grid_num = int(start_boundary * 2 / cell_size)
    grid = np.zeros(shape=(grid_num,grid_num))
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_agent_grid = []
    one_starts_landmark_grid = []
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
        for i in range(now_agent_num):
            while 1:
                landmark_location_grid = np.random.randint(0, grid.shape[0], 2) 
                if grid[landmark_location_grid[0],landmark_location_grid[1]]==1:
                    continue
                else:
                    grid[landmark_location_grid[0],landmark_location_grid[1]] = 1
                    one_starts_landmark_grid.append(copy.deepcopy(landmark_location_grid))
                    landmark_location = np.array([(landmark_location_grid[0]+0.5)*cell_size,(landmark_location_grid[1]+0.5)*cell_size])-start_boundary
                    one_starts_landmark.append(copy.deepcopy(landmark_location))
                    break
        # select_starts.append(one_starts_agent+one_starts_landmark)
        archive.append(one_starts_agent+one_starts_landmark)
        grid = np.zeros(shape=(grid_num,grid_num))
        one_starts_agent = []
        one_starts_agent_grid = []
        one_starts_landmark_grid
        one_starts_landmark = []
    return archive

def produce_uniform_case_grid_pb(num_case, start_boundary, now_agent_num, now_box_num):
    # agent_size=0.2, ball_size=0.2,landmark_size=0.3
    # box在内侧，agent在start_boundary和start_boundary_agent之间
    cell_size = 0.2
    grid_num = int((start_boundary[1]-start_boundary[0]) / cell_size) + 1
    init_origin_node = np.array([start_boundary[0],start_boundary[2]])
    assert grid_num ** 2 >= now_agent_num + now_box_num
    grid = np.zeros(shape=(grid_num,grid_num))
    one_starts_landmark = []
    one_starts_agent = []
    one_starts_box = []
    one_starts_box_grid = []
    one_starts_landmark_grid = []
    one_starts_agent_grid = []
    archive = [] 
    for j in range(num_case):
        for i in range(now_box_num):
            while 1:
                landmark_location_grid = np.random.randint(0, grid.shape[0], 2) 
                if grid[landmark_location_grid[0],landmark_location_grid[1]]==1:
                    continue
                else:
                    grid[landmark_location_grid[0],landmark_location_grid[1]] = 1
                    # box_location = np.array([(box_location_grid[0]+0.5)*cell_size,(box_location_grid[1]+0.5)*cell_size])-start_boundary
                    landmark_location = np.array([(landmark_location_grid[0]+0.5)*cell_size,(landmark_location_grid[1]+0.5)*cell_size]) + init_origin_node
                    one_starts_landmark.append(copy.deepcopy(landmark_location))
                    one_starts_landmark_grid.append(copy.deepcopy(landmark_location_grid))
                    break
        for i in range(now_box_num):
            while 1:
                box_location_grid = np.random.randint(0, grid.shape[0], 2) 
                if grid[box_location_grid[0],box_location_grid[1]]==1:
                    continue
                else:
                    grid[box_location_grid[0],box_location_grid[1]] = 1
                    # box_location = np.array([(box_location_grid[0]+0.5)*cell_size,(box_location_grid[1]+0.5)*cell_size])-start_boundary
                    box_location = np.array([(box_location_grid[0]+0.5)*cell_size,(box_location_grid[1]+0.5)*cell_size]) + init_origin_node
                    one_starts_box.append(copy.deepcopy(box_location))
                    one_starts_box_grid.append(copy.deepcopy(box_location_grid))
                    break
        for i in range(now_agent_num):
            agent_location = np.random.uniform(start_boundary[0], start_boundary[1], 2)
            one_starts_agent.append(copy.deepcopy(agent_location))
        # select_starts.append(one_starts_agent+one_starts_landmark)
        archive.append(one_starts_agent+one_starts_box+one_starts_landmark)
        grid = np.zeros(shape=(grid_num,grid_num))
        one_starts_agent = []
        one_starts_landmark = []
        one_starts_box = []
        one_starts_box_grid = []
        one_starts_landmark_grid = []
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
   
    # actor_critic = torch.load('/home/chenjy/curriculum/results/MPE/simple_spread/check/run10/models/agent_model.pt')['model'].to(device)
    actor_critic = torch.load('/home/tsing73/curriculum/results/MPE/50agent_model.pt')['model'].to(device)
    # filename = '/home/tsing73/curriculum/'
    # for name, para in zip(actor_critic.actor_base.state_dict(),actor_critic.actor_base.parameters()):
    #     pdb.set_trace()
    #     a = para.transpose(0,1).reshape(1,-1)
    #     np.savetxt(filename+ name + '.txt',np.array(a.to('cpu').detach()),delimiter=',\n')
    # pdb.set_trace()
    actor_critic.agents_num = args.num_agents
    actor_critic.boxes_num = args.num_boxes
    num_agents = args.num_agents
    num_boxes = args.num_boxes
    all_frames = []
    cover_rate = 0
    random.seed(1)
    np.random.seed(1)
    
    # load files
    dir_path = '/home/chenjy/curriculum/diversified_left/' + ('archive_' + str(89))
    if os.path.exists(dir_path):
        with open(dir_path,'r') as fp :
            archive = fp.readlines()
        for i in range(len(archive)):
            archive[i] = np.array(archive[i][1:-2].split(),dtype=float)

    # starts = produce_good_case_grid_pb(500,[-0.6,0.6,-0.6,0.6],num_agents,num_boxes)
    for eval_episode in range(args.eval_episodes):
        print(eval_episode)
        eval_env = MPEEnv(args)
        if args.save_gifs:
            image = eval_env.render('rgb_array', close=False)[0]
            all_frames.append(image)
        
        # eval_obs, _ = eval_env.reset(num_agents,num_boxes)
        eval_obs, _ = eval_env.reset(num_agents)
        # eval_obs = eval_env.new_starts_obs(start,num_agents)
        # eval_obs = eval_env.new_starts_obs_pb(starts[eval_episode],num_agents,num_boxes)
        eval_obs = np.array(eval_obs)       
        eval_share_obs = eval_obs.reshape(1, -1)
        eval_recurrent_hidden_states = np.zeros((num_agents,args.hidden_size)).astype(np.float32)
        eval_recurrent_hidden_states_critic = np.zeros((num_agents,args.hidden_size)).astype(np.float32)
        eval_masks = np.ones((num_agents,1)).astype(np.float32)
        step_cover_rate = np.zeros(shape=(args.episode_length))
        
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
            step_cover_rate[step] = eval_infos[0]['cover_rate']
            eval_obs = np.array(eval_obs)
            eval_share_obs = eval_obs.reshape(1, -1)
            
            if args.save_gifs:
                image = eval_env.render('rgb_array', close=False)[0]
                all_frames.append(image)
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < args.ifi:
                    time.sleep(ifi - elapsed)
        print('cover_rate: ', np.mean(step_cover_rate[-5:]))
        cover_rate += np.mean(step_cover_rate[-5:])                     
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
