- [scenariors](#scenariors)
- [world源码](#world源码)
- [Agent部分](#agent部分)
- [simpleEnv简单环境设定](#simpleenv简单环境设定)
## scenariors
```
class Scenario(BaseScenario):
    # 具体作用：配置各类别智能体个数、属性，同时创建障碍物
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2):
        world = World()
        # set any world properties first
        world.dim_c = 2

        # 好的agent数量:num_good
        # 敌对agent数量:num_adversaries
        # 总的agent数量:num_agents
        # 障碍物landmarks数量:num_obstacles
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        
        # 先创建num_adversaries个敌对agent，然后再创建好的agent，属性上也是从这里定义的
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3

        # 创建landmarks模块
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    # 初始化各个智能体以及障碍物的颜色、状态等
    def reset_world(self, world, np_random):
        # 将智能体根据其类别设定颜色
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            
        # 给障碍物填充颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            
        # 初始化智能体的状态，位置设为-1~1随机分布，维度为2维，速度初始化为0，交流初始化为0.
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)        
        
        # 初始化障碍物的状态，位置设为--0.9~0.9随机分布，维度为2维，速度初始化为0，交流初始化为0.
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    # 用来判断智能体是否为敌对智能体，如果是的话返回其与好的智能体碰撞的个数，如果不是返回0.
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

    # 判断两个智能体是否碰撞，碰撞判断的方法是用两者的距离与二者半径之和作比较，如果距离小于半径之和返回True，否则返回False
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    # 返回好的智能体的列表
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    # 返回敌对智能体的列表
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # 分别返回各个智能体的回报值
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    # 计算好的agent的reward
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            # 另一种形式的reward，采用的是增加reward的方式来进行相应的求值，与adv距离的欧氏距离*0.1作为reward的增加值。
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        
        # 采用的是降低reward的方式来进行agent的reward获取，如果与adv碰撞了，那么回报值-10。
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # 超出边界的处理方法：具体分析之后会将图片传入wiki
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    # 敌对agent的奖励值
    # 计算的是所有的敌对agent的奖励值之和，具体的奖励值与好的agent是刚好相反的
    # 这里传入agent参数的目的是确定当前任务是需要考虑碰撞的因素的，如果agent.collide=False就不把这个作为reward
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew


    # 
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []

        # 获取智能体agent与每个障碍物的距离
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []

        # 将其余agent与当前gaent之间的交流、距离元组、非敌对agent的速度分别保存到三个列表中
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        # 将当前agent的速度、位置 与 智能体agent与每个障碍物的距离 与 当前agent与其余各个agent(包括敌对agent)的距离元组 与 非敌对agent的速度
        # 放置到同样的一个ndarray中
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )

```
## world源码
```python
class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3

    # 获取所有智能体以及landmarks的列表
    @property
    def entities(self):
        return self.agents + self.landmarks

    # 策略智能体，不采用callback函数进行动作决策，直接通过策略网络获取动作
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # 脚本智能体，采用callback函数直接获取动作
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # world单步迭代
    def step(self):
        # 采用脚本对智能体进行移动
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # 将力进行集成，顺序分别为给动作添加u_noise、碰撞力、集成环境的力
        p_force = [None] * len(self.entities)
        p_force = self.apply_action_force(p_force)
        p_force = self.apply_environment_force(p_force)
        self.integrate_state(p_force)
                
        # 将每个智能体添加上c_noise
        for agent in self.agents:
            self.update_agent_state(agent)

    # 给每个动作添加动作噪声
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force

    # 集成环境碰撞带来的力，用于进一步更新状态
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

    # 集成状态，用于修改位置、速度
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )

    # 给动作加上交流噪声
    def update_agent_state(self, agent):
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # 碰撞力的记录，用于之后进行碰撞后的位移
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

```
## Agent部分
```
class EntityState:
    # 智能体的位置和速度
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(
    EntityState
): 
    # 智能体之间的交流(以speaker为例，就是发声)
    def __init__(self):
        super().__init__()
        self.c = None


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity: 
    # 智能体名字、尺寸、是否可以移动、是否可以碰撞、物体目睹、颜色、最大速度、加速度、状态、初始质量
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    # 返回初始质量
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    # 智能体的可移动性√，禁止交流×，不能观测到世界×，物理噪声None，交流噪声None，控制范围1，动作召回
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
```
## simpleEnv简单环境设定
```
class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }
    '''
    需要传入的参数：
        @param: 具体的情景——scenario
        @param: 世界环境——world
        @param: 每轮最大迭代次数——max_cycles
        @param: 表现方式——render_mode
        @param: 是否采用连续动作——continuous_actions
        @param: 本地比例,即每隔智能体的reward中自己产生的reward的比例,最后会与全局reward进行一次加权求和:      
            local_ratio*local_reward+(1-local_ratio)*global_reward
        ——local_ratio

    类内变量解析：
        self.render_mode:表述模式,对应["human", "rgb_array"],具体区别就是,前者是直接进行图像展示,后者是保存一个rgb图像array,需要自己采用plt展示
        self.viewer:暂不清楚
        self.width:render的图像宽度
        self.height:render的图像高度
        self.screen:基本的surface界面
        self.max_size:最大的尺寸
        self.game_font:render时界面所采取的字体
        self.renderOn:是否开启render
        self._seed():设定种子

        self.max_cycles:单次render最大步长数
        self.scenario:情景设置
        self.world:世界设置
        self.continuous_actions:是否采用连续空间
        self.local_ratio:本地比例

        self.agents:所有智能体的名字
        self.possible_agents:所有可能的智能体的名字
        self._index_map:智能体名字对应的下标的词典  i.e. 'agent_1':0 'agent_2':1
        self._agent_selector = agent_selector(self.agents)
        self.action_spaces:智能体的动作空间词典,具体用法self.action_spaces[agent.name]            
        self.observation_spaces:智能体的观测空间词典,具体用法self.observation_spaces[agent.name]
        self.state_space:状态空间,包含了所有智能体的观测维度
        self.steps:当前轮步长的序号
        self.current_actions:每个智能体当前所采取的动作

        self.np_random:随机数调用函数，可通过类似于self.np_random.random()的方式来调用

    '''
    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents

    # 获取每个智能体的观测空间
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # 获取每个智能体的动作空间
    def action_space(self, agent):
        return self.action_spaces[agent]

    # 获取随机数生成器self.np_random以及随机数种子seed
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    # 获取指定智能体在scenario下的具体观测值
    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    # 获取所有可能的智能体在环境下的观测值元组
    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    # 将奖励函数值以及各种变量进行初始化
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    # 先将动作设定到指定的各个agent上，进行一次世界移动后计算全局reward，并对每个智能体求一个全局+局部的reward
    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # 为指定agent设定动作，修改agent对应的action，通过修改agent实例类方法实现
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # 设定加速度，加速度的范围值根据agent.accel设定，如果参数没有定义的话那么默认为5.
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]

        # 设定交流动作
        if not agent.silent:
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        assert len(action) == 0

    # 进行单步迭代
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        # 参数的类型为str，表示智能体的ID
        # 循环对智能体进行状态迭代
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()
        self.current_actions[current_idx] = action

        # next_idx==0说明已经对每个智能体进行了一次相应动作的移动，将动作设定到智能体上，并在self.steps大于最大的移动步长时进行截断
        # next_idx!=0说明还没对每个智能体设定好动作，将reward值每次清空
        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        # 将累计回报清空并重新进行累计求和
        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    # 创建窗口并启动render
    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            # 创建一个大小与前面设定的screen大小一样的窗口，用来放置之前的scale
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    # 进行智能体的render操作
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            # 将pygame绘图保存为rgb_array的格式，并逐帧返回
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            # 将绘图进行刷新显示
            pygame.display.flip()
            return

    # 绘制每个step下的pygame图像
    def draw(self):
        # 背景设置为白色
        self.screen.fill((255, 255, 255))

        # 将每个entities的坐标记录，并求出最大的坐标位置
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # 获取坐标，并将y轴翻转，与原本pyglet坐标系相对应
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            
            # 将位置坐标进行归一化操作，防止出现过于远离边界
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            # 绘制圆形以及圆形的边界线
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # 书写对应的文字到 (0,0) 坐标位置
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    # 关闭render相关设定
    def close(self):
        if self.renderOn:
            # 将剩余事件处理
            pygame.event.pump()
            # 退出pygame展示
            pygame.display.quit()
            self.renderOn = False
```
