import numpy as np

from Ship_world import World
from Ship_ord_obj import Agent


class Scenario:
    def make_world(self):
        world = World()
        # TODO 进行agent类的命名等工作
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            # TODO 为每个智能体的其余性质进行赋值，目前暂未设定
            # i.e.
            # agent.collide = False
            # agent.silent = True
        
        return world

    def reset_world(self, world, np_random):
        # TODO 设定同类智能体的颜色，需要添加if语句
        for i, agent in enumerate(world.agents):
            # i.e.
            agent.color = np.array([0.25, 0.25, 0.25])

        # TODO 设定初始状态
        for agent in world.agents:
            # TODO 这部分老师说手工设定，不能随机设，仅做demo时使用这部分
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)

            # TODO 这部分需要设定一个列表值，假设单位为节，红A：6，红B,b，蓝A：40，其余情况需要另设
            # QUESTION 先确定需要设计成什么单位，再修改这部分,动作空间需不需要设定加速度与角度？
            agent.state.p_vel = np.zeros(world.dim_p)

    # TODO 设定奖励函数值，这里暂时不确定需要的全部奖励
    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    # TODO 返回相对位置
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
