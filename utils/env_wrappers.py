"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

def simplifyworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if all(done):
                    ob = env.reset()           
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()         
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SimplifySubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=simplifyworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd[0] == 'step':
            # import time; start = time.time()
            actions = cmd[1]
            now_agent_num = cmd[2]
            ob, reward, done, info, available_actions = env.step(actions)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, available_actions = env.reset(now_agent_num)
            else:
                if all(done):
                    ob, available_actions = env.reset(now_agent_num) 
            remote.send((ob, reward, done, info, available_actions))
        elif cmd[0] == 'reset':
            now_agent_num = cmd[1]
            ob, available_actions= env.reset(now_agent_num)
            remote.send((ob, available_actions))
        elif cmd[0] == 'reset_pb':
            now_agent_num = cmd[1]
            now_box_num = cmd[2]
            ob, available_actions= env.reset(now_agent_num,now_box_num)
            remote.send((ob, available_actions))
        elif cmd[0] == 'new_starts_obs':
            now_agent_num = cmd[1]
            starts_one = cmd[2]
            ob = env.new_starts_obs(starts_one,now_agent_num)
            remote.send(ob)
        elif cmd[0] == 'new_starts_obs_pb':
            now_agent_num = cmd[1]
            now_box_num = cmd[2]
            starts_one = cmd[3]
            ob = env.new_starts_obs_pb(starts_one,now_agent_num,now_box_num)
            remote.send(ob)
        # elif cmd[0] == 'reset_pb':
        #     now_agent_num = cmd[1]
        #     now_box_num = cmd[2]
        #     ob, available_actions= env.reset(now_agent_num,now_box_num)
        #     remote.send((ob, available_actions))

        # if cmd == 'step':
        #     ob, reward, done, info, available_actions = env.step(data)
        #     if done.__class__.__name__=='bool':
        #         if done:
        #             ob, available_actions = env.reset()
        #     else:
        #         if all(done):
        #             ob, available_actions = env.reset()
            
        #     remote.send((ob, reward, done, info, available_actions))
        # elif cmd == 'reset':
        #     ob, available_actions = env.reset()           
        #     remote.send((ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        self.length = len(env_fns)
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    # def step_async(self, actions):
    #     for remote, action in zip(self.remotes, actions):
    #         remote.send(('step', action))
    #     self.waiting = True

    # def step_wait(self):
    #     results = [remote.recv() for remote in self.remotes]
    #     self.waiting = False
    #     obs, rews, dones, infos, available_actions = zip(*results)
    #     return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def step_async(self, actions, now_num_processes, now_agent_num):
        i = 0
        for remote, action in zip(self.remotes, actions):
            tmp_list = ['step', action, now_agent_num]
            if i < now_num_processes:
                remote.send((tmp_list,None))
                i += 1
        self.waiting = True

    def step_wait(self,now_num_processes):
        results = []
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        self.waiting = False
        obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    # def reset(self):
    #     for remote in self.remotes:
    #         remote.send(('reset', None))
    #     results = [remote.recv() for remote in self.remotes]
    #     obs, available_actions = zip(*results)
    #     return np.stack(obs), np.stack(available_actions)

    # def reset(self, now_agent_num, now_box_num=None):
    #     if now_box_num is None:
    #         for remote in self.remotes:
    #             remote.send(('reset' + str(now_agent_num), None))
    #     else:
    #         for remote in self.remotes:
    #             remote.send(('reset' + str(now_agent_num), None))
    #     results = [remote.recv() for remote in self.remotes]
    #     obs, available_actions = zip(*results)
    #     self.remotes[0].send(('get_spaces', None))
    #     observation_space, action_space = self.remotes[0].recv()
    #     VecEnv.__init__(self, self.length, observation_space, action_space)
    #     return np.stack(obs), np.stack(available_actions)

    def reset(self, now_agent_num, now_box_num=None):
        if now_box_num is None:
            for remote in self.remotes:
                remote.send((['reset',now_agent_num], None))
        else:
            for remote in self.remotes:
                remote.send((['reset_pb',now_agent_num, now_box_num], None))
        results = [remote.recv() for remote in self.remotes]
        obs, available_actions = zip(*results)
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, self.length, observation_space, action_space)
        return np.stack(obs), np.stack(available_actions)

    def new_starts_obs(self, starts, now_agent_num, now_num_processes):
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                tmp_list = ['new_starts_obs', now_agent_num, starts[i]]
                remote.send((tmp_list, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, self.length, observation_space, action_space)
        return np.stack(results)

    def new_starts_obs_pb(self, starts, now_agent_num, now_box_num, now_num_processes):
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                tmp_list = ['new_starts_obs_pb', now_agent_num, now_box_num, starts[i]]
                remote.send((tmp_list, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, self.length, observation_space, action_space)
        return np.stack(results)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info, available_actions = env.step(data)            
            remote.send((ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, available_actions = env.reset(data)           
            remote.send((ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ChooseSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes,reset_choose):
            remote.send(('reset', choose))
        results = [remote.recv() for remote in self.remotes]
        obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
        
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos, available_actions = map(np.array, zip(*results))
        self.ts += 1
        
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], available_actions[i] = self.envs[i].reset()                   
                    self.ts[i] = 0
            else:
                if all(done):
                    obs[i], available_actions[i] = self.envs[i].reset()
                    self.ts[i] = 0
        
        self.actions = None

        return np.array(obs), np.array(rews), np.array(dones), infos, np.array(available_actions)

    def reset(self):  
        obs = []
        available_actions = []
        for env in self.envs:
            o,s = env.reset()
            obs.append(o)
            available_actions.append(s) 
        return np.array(obs), np.array(available_actions)

    def close(self):
        for env in self.envs:
            env.close()        