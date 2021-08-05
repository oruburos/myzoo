import numpy as np
#from pettingzoo.mpe._mpe_utils.core import World, Agent, Landmark
from Entities import World, Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
#from pettingzoo.mpe._mpe_utils.simple_env import  make_env ,SimpleEnv
from SimpleGerryEnv import SimpleGerryEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.rendering import Viewer



class GerryScenario(BaseScenario):
    def make_world(self, num_good_agents=0, num_adversaries=1, num_landmarks=30, num_food=0, num_forests=0):
        world = World()
        # set any world properties first
        world.dim_c = 4

        num_good_agents = 1
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents
        #num_landmarks = num_landmarks
        num_landmarks = 3
      #  print("current landmark", str(num_landmarks))

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_index = i - 1 if i < num_adversaries else i - num_adversaries
            base_index = 0 if base_index < 0 else base_index
            base_name = "adversary" if agent.adversary else "agent"
            base_name = "leadadversary" if i == 0 else base_name
            agent.name = '{}_{}'.format(base_name, base_index)
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False
            agent.size = 0.1
            agent.accel = 1
            agent.max_speed = 0.001

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            # print("current landmark", str(i))
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = True
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def checkDistricts(self, world, np_random):
        for i, landmark in enumerate(world.landmarks):
            #print("current landmark", str(i))
            landmark.color = np.array([0.25, 0.25, 0.25])
            # landmark.state.p_pos = [i*10,i*10]
            landmark.state.p_pos = np_random.uniform(-5, +5, world.dim_p)


        # assuming 3 districts
        print("checking locations")
        if (self.is_collision( world.landmarks[0] ,world.landmarks[1]  )):
            print("regenerate district 1")
        elif (self.is_collision(world.landmarks[2], world.landmarks[1])):
            print("regenerate district 2")





    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([1, 0, 0]) if not agent.adversary else np.array([0., 0, 1])
            # random properties for landmarks

        self.checkDistricts(world, np_random)
        for agent in world.agents :
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-0.2, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)



    def benchmark_data(self, agent, world):
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
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False

    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 5

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 2 * bound(x)


        return rew

    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 5
        return rew

    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)


        # to tell the pred when the prey are in the forest

        ga = self.good_agents(world)


        comm = [world.agents[0].state.c]

        if agent.adversary and not agent.leader:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + comm)
        if agent.leader:
            return np.concatenate(
                [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel  + comm)
        else:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)


class raw_env(SimpleGerryEnv):
    def __init__(self, num_good=2, num_adversaries=4, num_obstacles=1, num_food=2, max_cycles=25, num_forests=2, continuous_actions=False):
        scenario = GerryScenario()
        print("after creating GerryMandering scenario")
        world = scenario.make_world(num_good, num_adversaries, num_obstacles, num_food, num_forests)
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "GerryEnvironment"
env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)