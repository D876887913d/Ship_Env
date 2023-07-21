import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        
        # 世界维度设定为3维，三个landmark，可以协作
        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True

        # 设定两个智能体的具体信息
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
            
        # speaker不可移动
        world.agents[0].movable = False

        # listener不可发送消息
        world.agents[1].silent = True

        # 设定landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04

        # 重置对应的状态等
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 为智能体设定两个目标
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        
        # 设定speaker的目标a为智能体listener，目标b为任一landmarks
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        
        # 设定智能体的颜色
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])               
        
        # 设定各个landmarks的颜色
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])

        # 将目标a的颜色设定为目标b的颜色加上rgb(0.45,0.45,0.45)
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])

        # 设定坐标、速度、交流
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # 设定各个landmarks的位置、速度
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # 基准数据
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return self.reward(agent, agent,world)

    # 计算
    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    # 返回目标的颜色
    def observation(self, agent, world):
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # 获取智能体与landmarks的距离
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # 将comm添加上其他智能体的交流数据
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)
        
        # 如果是speaker的话，返回目标颜色
        if not agent.movable:
            return np.concatenate([goal_color])
        
        # 如果是listener的话，返回智能体的速度、位置、交流数据
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
            
