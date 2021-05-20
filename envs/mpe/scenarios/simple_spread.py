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
            landmark.cover = 0
            # landmark.size = 0.15
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
    
    def landmark_cover_state(self, world):
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) <= world.agents[0].size + world.landmarks[0].size:
                l.cover = 1
            else:
                l.cover = 0

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
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a!=agent:
                    rew -= 1
                    break
        return 0.1*rew

    def get_info(self, world):
        # cover
        num = 0
        success = False
        entity_cover_state = []
        infos = {}
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) <= world.agents[0].size + world.landmarks[0].size:
                num = num + 1
                entity_cover_state.append(1)
            else:
                entity_cover_state.append(0)
        # success
        if num==len(world.landmarks):
            success = True
        # collision
        collision_num = 0
        for agent in world.agents:
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent) and a!=agent:
                        collision_num += 1
                        break
        
        # position info
        pos_info = []
        for agent in world.agents:
            pos_info.append(agent.state.p_pos)
        for landmark in world.landmarks:
            pos_info.append(landmark.state.p_pos)
        info_list = {'cover_rate': num/len(world.landmarks),'collision': collision_num, 'success': success, 'pos_state': np.array(pos_info)}
        return info_list

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            tmp_pos = np.insert((entity.state.p_pos - agent.state.p_pos),0,entity.cover)
            entity_pos.append(tmp_pos) # 是否被cover
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

    def get_state(self, world):
        pass

    def share_reward(self, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        cover_num = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < world.agents[0].size + world.landmarks[0].size:
                # rew += 8/len(world.agents)
                cover_num += 1
        if cover_num == len(world.agents):
            rew += 4
        return 0.1*rew
