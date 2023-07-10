import numpy as np

class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        
        # 表示坐标维度为二维，x,y
        self.dim_p = 2
        
        # 总共有两种不同颜色的单元，蓝色以及红色
        self.dim_color = 2

        # simulation timestep
        self.dt = 0.1

        # # physical damping
        # TODO 由于忽略阻力，假设物理阻尼为0，之后可以根据需要设定
        # self.damping = 0

        # # 智能体之间不存在交流，故下面这些参数暂且全赋值为0
        # # communication channel dimensionality
        # self.dim_c = 0

        # TODO 根据对于碰撞力的需求，以及碰撞实际的效果，设定碰撞相关参数
        # # contact response parameters
        # self.contact_force = 0
        # self.contact_margin = 0

    # return all entities in the world
    # 将所有的智能体总列表返回
    @property
    def entities(self):
        return self.agents

    # return all agents controllable by external policies
    # 返回所有依靠策略的智能体
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    # 返回所有依靠设定好的脚本的智能体
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # 环境进行单步更新
    # update state of the world
    def step(self):
        # set actions for scripted agents
        # 采用已有脚本控制非智能体
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
            
        # gather forces applied to entities
        # TODO 将一些外界力作用进行整合，在要做的航船环境中应当修改为 角度force 以及 加速度force.
        p_force = [None] * len(self.entities)
        
        # 添加噪声
        p_force = self.apply_action_force(p_force)

        # apply environment forces
        # TODO 判断智能体是否碰撞，并保存碰撞后的力,待修改
        # p_force = self.apply_environment_force(p_force)

        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        # TODO 给原本的动作空间添加少许噪声，提高模型泛化能力
        # 动作空间的操作需要进行大幅修改，参见Ship_ord_obj的Agent类
        for i, agent in enumerate(self.agents):
            noise = (
                np.random.randn(*agent.action.u.shape) * agent.u_noise
                if agent.u_noise
                else 0.0
            )
            p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    # TODO collision相关函数，force部分待修改
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    # TODO 处理超过最大速度、最大边界时的情况
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            # 新的位置=当前速度*时间间隔
            entity.state.p_pos += entity.state.p_vel * self.dt

            # 新的速度=当前速度+加速度
            entity.state.p_vel += entity.action.aclr

            # TODO 根据给定的action space force确定对应的速度变化方式，即进行速度、坐标等状态的变化
            # 这步实现很重要！
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt

            if entity.max_speed is not None:
                speed = entity.state.p_vel

                # 速度截断
                # 由于仅有单个方向的速度，因此化简起来较为简单，即直接把最大值作为当前速度
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.max_speed

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        # TODO 如果存在状态联系，那么对智能体进行如下操作
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    # 碰撞力的设定，仅适用于可进行移动的智能体
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        
        # 计算两坐标之间的欧氏距离
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        
        # minimum allowable distance
        # 计算两个智能体不相互覆盖的最小距离
        dist_min = entity_a.size + entity_b.size
        
        # softmax penetration
        # TODO 接触间隔，用于优化具体的action force配置
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k

        # TODO 配置利用加速度、角度变换进行的位置变换，预计为两个智能体的两个动作[[a_1,r_1],[a_2,r_2]]
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
