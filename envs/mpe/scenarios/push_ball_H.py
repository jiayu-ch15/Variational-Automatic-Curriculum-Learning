import numpy as np
from envs.mpe.core import World, Agent, Landmark, Wall
from envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args, now_agent_num=None, now_box_num=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        if now_agent_num==None:
            num_people = args.num_agents
            num_boxes = args.num_landmarks
            num_landmarks = args.num_landmarks
        else:
            num_people = now_agent_num
            num_boxes = now_box_num
            num_landmarks = now_box_num
        self.num_boxes = num_boxes
        self.num_people = num_people
        self.num_agents = num_boxes + num_people # deactivate "good" agent
        # add walls 6*2
        world.walls = [Wall(orient='V', axis_pos=-4.5, endpoints=(-6, 6),width=3.0, hard=True),
                Wall(orient='V', axis_pos=4.5, endpoints=(-6, 6),width=3.0, hard=True),
                Wall(orient='H', axis_pos=-4, endpoints=(-6, 6),width=3.0, hard=True),
                Wall(orient='H', axis_pos=4, endpoints=(-6, 6),width=3.0,hard=True),
                # up left wall
                Wall(orient='V', axis_pos=0, endpoints=(0.2, 3), width=2, hard=True),
                # down left wall
                Wall(orient='V', axis_pos=0, endpoints=(-3, -0.2), width=2, hard=True),
                ]
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_people else False  # people.adversary = True     box.adversary = False
            agent.size = 0.1 if agent.adversary else 0.15
            # agent.accel = 3.0 if agent.adversary else 5
            # agent.max_speed = 0.5 if agent.adversary else 0.5
            agent.action_callback = None if i < num_people else self.box_policy  # box有action_callback 即不做动作

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15
            landmark.cover = 0
            # landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def box_policy(self, agent, world):
        chosen_action = np.array([0,0], dtype=np.float32)
        # chosen_action_c = np.array([0,0], dtype=np.float32)
        return chosen_action

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0, 0, 0])
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-3.0, +3.0, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3.0, +3.0, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def landmark_cover_state(self, world):
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) <= world.agents[0].size + world.landmarks[0].size:
                l.cover = 1
            else:
                l.cover = 0

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.adversary == False]  # 算box和landmark间的距离
        #     if min(dists) < world.landmarks[0].size + world.agents[-1].size:
        #         rew += 1/self.num_people

        if agent.collide and not agent.adversary:
            for a in world.agents:
                if self.is_collision(a, agent) and not a.adversary:
                    if a!=agent:
                        rew -= 1.0
                        break
        return 0.1*rew
    
    def reset_radius(self,sample_radius):
        sample_radius = max(sample_radius,1.5)
        self.sample_radius = sample_radius

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        landmark_pos = []
        for entity in world.landmarks:  # world.entities:
            tmp_pos = np.insert((entity.state.p_pos - agent.state.p_pos),0,entity.cover)
            landmark_pos.append(tmp_pos) # 是否被cover
        # communication of all other agents
        comm = []
        adv_pos = []
        good_pos = []
        for other in world.agents:
            if other is agent: continue
            if other.adversary:
                adv_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                good_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + adv_pos + good_pos + landmark_pos)
    
    def info_coverage_rate(self, world):
        num = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.adversary == False]
            if min(dists) <= world.agents[-1].size + world.landmarks[0].size:
                num = num + 1
        return num/len(world.landmarks)

    def share_reward(self, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        cover_num = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.adversary == False]
            # rew -= min(dists)
            if min(dists) <= world.landmarks[0].size + world.agents[-1].size:
                rew += 2.0/self.num_people
                cover_num += 1
        if cover_num == self.num_boxes:
            rew += 1.0
        return 0.1*rew
