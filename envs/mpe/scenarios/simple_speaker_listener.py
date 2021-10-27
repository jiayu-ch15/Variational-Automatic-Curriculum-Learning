import numpy as np
from envs.mpe.core import World, Agent, Landmark
from envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args, now_agent_num=None):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        num_agents = 2
        assert num_agents==2, ("only 2 agents is supported, check the config.py.")
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])               
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        reward = 0
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))

        # sparse reward
        cover_num = 0
        if dist2 < world.agents[0].size + world.landmarks[0].size:
            cover_num += 1
        if cover_num == 1:
            reward += 1
        return 0.1 * reward

        # # dense reward
        # return -dist2

    def share_reward(self, world):
        return 0.0

    def landmark_cover_state(self, world):
        return None

    def get_state(self, world):
        pass

    def get_info(self, world):
        num = 0
        success = False
        entity_cover_state = []
        infos = {}
        
        # cover
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        num = 0.0
        if dist2 < world.agents[0].size + world.landmarks[0].size:
            num += 1.0
        success = False
        if num == 1:
            success = True
        
        # position info
        pos_info = []
        for agent in world.agents:
            pos_info.append(agent.state.p_pos)
        for landmark in world.landmarks:
            pos_info.append(landmark.state.p_pos)

        info_list = {'cover_rate': num, 'success': success, 'pos_state': np.array(pos_info), 'achieved_goal': world.agents[1].state.p_pos, 'goal_state': world.agents[0].goal_b.state.p_pos}
        return info_list

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)
        
        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
            
