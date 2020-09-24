import numpy as np
from envs.mpe.core import World, Agent, Landmark
from envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args, now_agent_num=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        if now_agent_num==None:
            num_agents = args.num_agents
            num_landmarks = args.num_landmarks
        else:
            num_agents = now_agent_num
            num_landmarks = now_agent_num
        # world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            # agent.size = 0.15
            agent.size = 0.1 
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            # landmark.size = 0.12
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3, +3, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-3, +3, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a!=agent:
                    # rew -= 1/len(world.agents)
                    rew -= 1
                    break
        return 0.1*rew

    def info_coverage_rate(self, world):
        num = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) <= world.agents[0].size + world.landmarks[0].size:
                num = num + 1
        return num/len(world.landmarks)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
    
    def share_reward(self, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        cover_num = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < world.agents[0].size + world.landmarks[0].size:
                rew += 8/len(world.agents)
                cover_num += 1
        # if cover_num == len(world.agents):
        #     rew += 5
        return 0.1*rew
