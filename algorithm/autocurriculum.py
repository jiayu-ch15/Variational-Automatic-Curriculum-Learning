import numpy as np
import random
import pdb
import wandb
from pathlib import Path
from scipy.spatial.distance import cdist
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from envs import MPEEnv
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.storage import RolloutStorage, RolloutStorage_share

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
        # pdb.set_trace()
        for j in range(num_case):
            for i in range(now_agent_num):
                landmark_location = np.array([np.random.uniform(start_boundary['x'][0],start_boundary['x'][1]),np.random.uniform(start_boundary['y'][0],start_boundary['y'][1])])
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
        # pdb.set_trace()
        return archive

    def uniform_sampling(self, starts_length, boundary, env_name):
        if env_name == 'simple_spread':
            one_starts_landmark = []
            one_starts_agent = []
            archive = [] 
            for j in range(starts_length):
                for i in range(self.num_agents): 
                    landmark_location = np.array([np.random.uniform(boundary['x'][0],boundary['x'][1]),np.random.uniform(boundary['y'][0],boundary['y'][1])])
                    one_starts_landmark.append(copy.deepcopy(landmark_location))
                    agent_location = np.array([np.random.uniform(boundary['x'][0],boundary['x'][1]),np.random.uniform(boundary['y'][0],boundary['y'][1])])
                    one_starts_agent.append(copy.deepcopy(agent_location))
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
                        if st[i][0] > self.boundary['x'][1]:
                            st[i][0] = self.boundary['x'][1] - random.random()*0.01
                        if st[i][0] < self.boundary['x'][0]:
                            st[i][0] = self.boundary['x'][0] + random.random()*0.01
                        if st[i][1] > self.boundary['y'][1]:
                            st[i][1] = self.boundary['y'][1] - random.random()*0.01
                        if st[i][1] < self.boundary['y'][0]:
                            st[i][1] = self.boundary['y'][0] + random.random()*0.01
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
                        if st[i][0] > self.boundary['x'][1]:
                            st[i][0] = self.boundary['x'][1] - random.random()*0.01
                        if st[i][0] < self.boundary['x'][0]:
                            st[i][0] = self.boundary['x'][0] + random.random()*0.01
                        if st[i][1] > self.boundary['y'][1]:
                            st[i][1] = self.boundary['y'][1] - random.random()*0.01
                        if st[i][1] < self.boundary['y'][0]:
                            st[i][1] = self.boundary['y'][0] + random.random()*0.01
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
        return starts, one_length, starts_length
    
    def move_nodes(self, one_length, Rmax, Rmin, del_switch, writer, timestep, inverted=False): 
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
                if inverted: # low value score means parents
                    if self.eval_score[i]<Rmin:
                        self.parent.append(copy.deepcopy(self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]))
                        del self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]
                        del_archive_num += 1
                else:
                    if self.eval_score[i]>Rmax:
                        self.parent.append(copy.deepcopy(self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]))
                        del self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]
                        del_archive_num += 1
        self.archive += child2archive
        self.parent_all += self.parent
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

    def move_nodes_value(self, one_length, value_threshold, del_switch, writer, timestep): 
        del_child_num = 0
        del_archive_num = 0
        del_easy_num = 0
        add_hard_num = 0
        drop_num = 0
        self.parent = []
        child2archive = []
        for i in range(one_length):
            if i < len(self.choose_child_index):
                if self.eval_score[i] >= value_threshold:
                    child2archive.append(copy.deepcopy(self.childlist[self.choose_child_index[i]-del_child_num]))
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
                else:
                    self.parent.append(copy.deepcopy(self.childlist[self.choose_child_index[i]-del_child_num]))
                    del self.childlist[self.choose_child_index[i]-del_child_num]
                    del_child_num += 1
            else:
                if self.eval_score[i]< value_threshold:
                    self.parent.append(copy.deepcopy(self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]))
                    del self.archive[self.choose_archive_index[i-len(self.choose_child_index)]-del_archive_num]
                    del_archive_num += 1

        self.archive += child2archive
        self.parent_all += self.parent
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

class goal_proposal():
    def __init__(self,num_agents, boundary, env_name, critic_k, buffer_capacity, proposal_batch):
        # TODO: zip env parameters
        self.num_agents = num_agents
        self.boundary = boundary
        self.env_name = env_name
        self.critic_k = critic_k
        self.buffer_capacity = buffer_capacity
        self.proposal_batch = proposal_batch

    def easy_sampling(self, starts_length, start_boundary):
        one_starts_landmark = []
        one_starts_agent = []
        archive = [] 
        for j in range(starts_length):
            for i in range(self.num_agents):
                landmark_location = np.array([np.random.uniform(start_boundary['x'][0],start_boundary['x'][1]),np.random.uniform(start_boundary['y'][0],start_boundary['y'][1])])
                one_starts_landmark.append(copy.deepcopy(landmark_location))
            indices = random.sample(range(self.num_agents), self.num_agents)
            for k in indices:
                epsilon = -2 * 0.01 * random.random() + 0.01
                one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+epsilon))
            # select_starts.append(one_starts_agent+one_starts_landmark)
            archive.append(one_starts_agent+one_starts_landmark)
            one_starts_agent = []
            one_starts_landmark = []
        return archive

    def uniform_sampling(self, starts_length, boundary):
        if self.env_name == 'simple_spread':
            one_starts_landmark = []
            one_starts_agent = []
            archive = [] 
            for j in range(starts_length):
                for i in range(self.num_agents): 
                    landmark_location = np.array([np.random.uniform(boundary['x'][0],boundary['x'][1]),np.random.uniform(boundary['y'][0],boundary['y'][1])])
                    one_starts_landmark.append(copy.deepcopy(landmark_location))
                    agent_location = np.array([np.random.uniform(boundary['x'][0],boundary['x'][1]),np.random.uniform(boundary['y'][0],boundary['y'][1])])
                    one_starts_agent.append(copy.deepcopy(agent_location))
                archive.append(one_starts_agent+one_starts_landmark)
                one_starts_agent = []
                one_starts_landmark = []
            return archive

    def value_disagreement_sampling(self, starts, actor_critic, starts_share_obs, starts_obs, starts_recurrent_hidden_states, starts_recurrent_hidden_states_critic, starts_masks):
        starts_value_list = np.zeros((self.buffer_capacity, self.num_agents, self.critic_k)).astype(np.float32)
        for agent_id in range(self.num_agents):
            actor_critic.eval() 
            starts_value,_,_ = actor_critic.get_value(
                                        torch.FloatTensor(starts_share_obs[:,agent_id]),
                                        torch.FloatTensor(starts_obs[:,agent_id]),
                                        torch.FloatTensor(starts_recurrent_hidden_states[:,agent_id]),
                                        torch.FloatTensor(starts_recurrent_hidden_states_critic[:,agent_id]),
                                        torch.FloatTensor(starts_masks[:,agent_id]))
            starts_value = starts_value.detach().cpu().numpy().squeeze(-1)
            starts_value_list[:,agent_id,:] = starts_value
        # deal with multi-agent, VDN or Q-mix
        starts_value_list = np.sum(starts_value_list,axis=1)
        # value std
        starts_value_list = np.std(starts_value_list,axis=1)
        average_value_disagreement = np.mean(starts_value_list)
        # probability
        starts_value_list = starts_value_list / np.sum(starts_value_list)
        starts_index = np.arange(0,self.buffer_capacity,1)
        # no repeat sampling
        starts_proposal_index = np.random.choice(starts_index,size=self.proposal_batch,replace=False,p=starts_value_list)
        starts_proposal = []
        for start_id in range(len(starts_proposal_index)):
            starts_proposal.append(starts[starts_proposal_index[start_id]])
        # end region
        return starts_proposal, average_value_disagreement
    
    def TD_error_sampling(self, args, starts, envs, actor_critic, starts_share_obs, starts_obs, starts_recurrent_hidden_states, starts_recurrent_hidden_states_critic, starts_masks):
        starts_value_list = np.zeros((self.buffer_capacity, self.num_agents, 2)).astype(np.float32)
        # get V(s0) and action
        actions= []
        recurrent_hidden_statess = []
        recurrent_hidden_statess_critic = []
        for agent_id in range(self.num_agents):
            actor_critic.eval()
            value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                torch.FloatTensor(starts_share_obs[:,agent_id]), 
                torch.FloatTensor(starts_obs[:,agent_id]), 
                torch.FloatTensor(starts_recurrent_hidden_states[:,agent_id]), 
                torch.FloatTensor(starts_recurrent_hidden_states_critic[:,agent_id]),
                torch.FloatTensor(starts_masks[:,agent_id]))
            value = value.detach().cpu().numpy().squeeze(-1)
            starts_value_list[:,agent_id,0] = value # get V(si)
            actions.append(action.detach().cpu().numpy())
            recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
            recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
        # rearrange action
        actions_env = []
        for i in range(self.buffer_capacity):
            one_hot_action_env = []
            for agent_id in range(self.num_agents):
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
        # get s1
        obs = []
        rewards = []
        for epoch_id in range(int(self.buffer_capacity/self.proposal_batch)):
            obs_epoch = envs.new_starts_obs(starts[epoch_id*self.proposal_batch:(epoch_id+1)*self.proposal_batch], self.num_agents, self.proposal_batch)
            obs_epoch, rewards_epoch, dones, infos, _ = envs.step(actions_env[epoch_id*self.proposal_batch:(epoch_id+1)*self.proposal_batch], self.proposal_batch, self.num_agents)
            obs.append(obs_epoch)
            rewards.append(rewards_epoch)
        obs = np.concatenate(obs,axis=0)
        rewards = np.concatenate(rewards,axis=0).squeeze(axis=2)
        share_obs = obs.reshape(self.buffer_capacity, -1)        
        share_obs = np.expand_dims(share_obs,1).repeat(self.num_agents,axis=1) 
        recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
        recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
        # get V(s1)
        for agent_id in range(self.num_agents):
            actor_critic.eval() 
            value,_,_ = actor_critic.get_value(agent_id,
                                        torch.FloatTensor(share_obs[:,agent_id]),
                                        torch.FloatTensor(obs[:,agent_id]),
                                        torch.FloatTensor(recurrent_hidden_states[:,agent_id]),
                                        torch.FloatTensor(recurrent_hidden_states_critic[:,agent_id]),
                                        torch.FloatTensor(starts_masks[:,agent_id]))
            value = value.detach().cpu().numpy().squeeze(-1)
            starts_value_list[:,agent_id,1] = value
        # get TD_error(0)
        TD_error = np.zeros((self.buffer_capacity, self.num_agents)).astype(np.float32)
        TD_error = np.abs(rewards + args.gamma * starts_value_list[:,:,1] - starts_value_list[:,:,0])

        # deal with multi-agent, VDN
        TD_error = np.sum(TD_error,axis=1)
        average_TD_error = np.mean(TD_error)
        # probability
        TD_error = TD_error / np.sum(TD_error)
        starts_index = np.arange(0,self.buffer_capacity,1)
        # no repeat sampling
        starts_proposal_index = np.random.choice(starts_index,size=self.proposal_batch,replace=False,p=TD_error)
        starts_proposal = []
        for start_id in range(len(starts_proposal_index)):
            starts_proposal.append(starts[starts_proposal_index[start_id]])
        # end region
        return starts_proposal, average_TD_error

def TD_error_score(args, batch_num, starts, envs, actor_critic, starts_share_obs, starts_obs, starts_recurrent_hidden_states, starts_recurrent_hidden_states_critic, starts_masks):
    starts_value_list = np.zeros((batch_num, args.num_agents, 2)).astype(np.float32)
    # get V(s0) and action
    actions= []
    recurrent_hidden_statess = []
    recurrent_hidden_statess_critic = []
    for agent_id in range(args.num_agents):
        actor_critic.eval()
        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
            torch.FloatTensor(starts_share_obs[:,agent_id]), 
            torch.FloatTensor(starts_obs[:,agent_id]), 
            torch.FloatTensor(starts_recurrent_hidden_states[:,agent_id]), 
            torch.FloatTensor(starts_recurrent_hidden_states_critic[:,agent_id]),
            torch.FloatTensor(starts_masks[:,agent_id]))
        value = value.detach().cpu().numpy().squeeze(-1)
        starts_value_list[:,agent_id,0] = value # get V(si)
        actions.append(action.detach().cpu().numpy())
        recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
        recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
    # rearrange action
    actions_env = []
    for i in range(batch_num):
        one_hot_action_env = []
        for agent_id in range(args.num_agents):
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
    # get s1
    obs = []
    rewards = []
    for epoch_id in range(int(batch_num/args.n_rollout_threads)):
        obs_epoch = envs.new_starts_obs(starts[epoch_id*args.n_rollout_threads:(epoch_id+1)*args.n_rollout_threads], args.num_agents, args.n_rollout_threads)
        obs_epoch, rewards_epoch, dones, infos, _ = envs.step(actions_env[epoch_id*args.n_rollout_threads:(epoch_id+1)*args.n_rollout_threads], args.n_rollout_threads, args.num_agents)
        obs.append(obs_epoch)
        rewards.append(rewards_epoch)
    obs = np.concatenate(obs,axis=0)
    rewards = np.concatenate(rewards,axis=0).squeeze(axis=2)
    share_obs = obs.reshape(batch_num, -1)        
    share_obs = np.expand_dims(share_obs,1).repeat(args.num_agents,axis=1) 
    recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
    recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
    # get V(s1)
    for agent_id in range(args.num_agents):
        actor_critic.eval() 
        value,_,_ = actor_critic.get_value(agent_id,
                                    torch.FloatTensor(share_obs[:,agent_id]),
                                    torch.FloatTensor(obs[:,agent_id]),
                                    torch.FloatTensor(recurrent_hidden_states[:,agent_id]),
                                    torch.FloatTensor(recurrent_hidden_states_critic[:,agent_id]),
                                    torch.FloatTensor(starts_masks[:,agent_id]))
        value = value.detach().cpu().numpy().squeeze(-1)
        starts_value_list[:,agent_id,1] = value
    # get TD_error(0)
    TD_error = np.zeros((batch_num, args.num_agents)).astype(np.float32)
    TD_error = np.abs(rewards + args.gamma * starts_value_list[:,:,1] - starts_value_list[:,:,0])

    # deal with multi-agent, VDN
    TD_error = np.sum(TD_error,axis=1)
    average_TD_error = np.mean(TD_error)
    # normalization
    TD_error_std = np.std(TD_error)
    TD_error_score = TD_error / TD_error_std
    average_TD_error_norm = np.mean(TD_error)
    return TD_error_score, average_TD_error, average_TD_error_norm

def value_error_score(args, batch_num, starts, envs, agents, actor_critic, use_gae, use_one_step):
    obs = envs.new_starts_obs(starts, args.num_agents, args.n_rollout_threads)
    #replay buffer
    rollouts = RolloutStorage(args.num_agents,
                args.episode_length, 
                args.n_rollout_threads,
                envs.observation_space[0], 
                envs.action_space[0],
                args.hidden_size) 
    # replay buffer init
    if args.share_policy: 
        share_obs = obs.reshape(args.n_rollout_threads, -1)        
        share_obs = np.expand_dims(share_obs,1).repeat(args.num_agents,axis=1)    
        rollouts.share_obs[0] = share_obs.copy() 
        rollouts.obs[0] = obs.copy()               
        rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
        rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    else:
        share_obs = []
        for o in obs:
            share_obs.append(list(itertools.chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(args.num_agents):    
            rollouts[agent_id].share_obs[0] = share_obs.copy()
            rollouts[agent_id].obs[0] = np.array(list(obs[:,agent_id])).copy()               
            rollouts[agent_id].recurrent_hidden_states = np.zeros(rollouts[agent_id].recurrent_hidden_states.shape).astype(np.float32)
            rollouts[agent_id].recurrent_hidden_states_critic = np.zeros(rollouts[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)

    for step in range(args.episode_length):
        # Sample actions
        values = []
        actions= []
        action_log_probs = []
        recurrent_hidden_statess = []
        recurrent_hidden_statess_critic = []
        
        with torch.no_grad():                
            for agent_id in range(args.num_agents):
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
        for i in range(args.n_rollout_threads):
            one_hot_action_env = []
            for agent_id in range(args.num_agents):
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
        obs, rewards, dones, infos, _ = envs.step(actions_env, args.n_rollout_threads, args.num_agents)
        
        # If done then clean the history of observations.
        # insert data in buffer
        masks = []
        for i, done in enumerate(dones): 
            mask = []               
            for agent_id in range(args.num_agents): 
                if done[agent_id]:    
                    recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                    recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                    mask.append([0.0])
                else:
                    mask.append([1.0])
            masks.append(mask)
                        
        if args.share_policy: 
            share_obs = obs.reshape(args.n_rollout_threads, -1)        
            share_obs = np.expand_dims(share_obs,1).repeat(args.num_agents,axis=1)    
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
            for agent_id in range(args.num_agents):
                rollouts[agent_id].insert(share_obs, 
                        np.array(list(obs[:,agent_id])), 
                        np.array(recurrent_hidden_statess[agent_id]), 
                        np.array(recurrent_hidden_statess_critic[agent_id]), 
                        np.array(actions[agent_id]),
                        np.array(action_log_probs[agent_id]), 
                        np.array(values[agent_id]),
                        rewards[:,agent_id], 
                        np.array(masks)[:,agent_id])
        # get value and compute return
        with torch.no_grad():
            for agent_id in range(args.num_agents):         
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
                                    use_gae, 
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
                                            use_gae, 
                                            args.gamma,
                                            args.gae_lambda, 
                                            args.use_proper_time_limits,
                                            args.use_popart,
                                            agents[agent_id].value_normalizer)
    value_error = np.abs(rollouts.returns[:,:] - rollouts.value_preds[step,:])
    # deal with multi-agent, VDN
    value_error = np.sum(value_error,axis=2)
    if use_one_step:
        value_error = value_error[0]
    else:
        value_error = np.mean(value_error,axis=0)
    average_value_error = np.mean(value_error)
    # normalization
    value_error_std = np.std(value_error)
    value_error_score = value_error / value_error_std
    average_value_error_norm = np.mean(value_error_score)
    return value_error_score, average_value_error, average_value_error_norm

def value_disagreement_score(args, batch_num, starts, actor_critic, starts_share_obs, starts_obs, starts_recurrent_hidden_states, starts_recurrent_hidden_states_critic, starts_masks):
    starts_value_list = np.zeros((batch_num, args.num_agents, args.critic_k)).astype(np.float32)
    for agent_id in range(args.num_agents):
        starts_value,_,_ = actor_critic.get_value(
                                    torch.FloatTensor(starts_share_obs[:,agent_id]),
                                    torch.FloatTensor(starts_obs[:,agent_id]),
                                    torch.FloatTensor(starts_recurrent_hidden_states[:,agent_id]),
                                    torch.FloatTensor(starts_recurrent_hidden_states_critic[:,agent_id]),
                                    torch.FloatTensor(starts_masks[:,agent_id]))
        starts_value = starts_value.detach().cpu().numpy().squeeze(-1)
        starts_value_list[:,agent_id,:] = starts_value
    # deal with multi-agent, VDN or Q-mix
    starts_value_list = np.sum(starts_value_list,axis=1)
    # value std
    starts_value_list = np.std(starts_value_list,axis=1)
    # normalization
    value_disagreement_std = np.std(starts_value_list)
    average_value_disagreement = np.mean(starts_value_list)
    disagreement_score = starts_value_list / value_disagreement_std
    average_value_disagreement_norm = np.mean(disagreement_score)
    return disagreement_score, average_value_disagreement, average_value_disagreement_norm

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
