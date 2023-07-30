class AgentState:  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # xy坐标值/速度
        self.p_pos=None
        self.p_vel=None


class Action:  # action of the agent
    def __init__(self):
        # 角度/加速度        
        self.agle=None
        self.aclr=None

'''
:breif 这个函数对应的是智能体的基本属性，例如坐标、速度、方向等，用于之后进行进一步的环境构建的时候进行配置各智能体。
:expand 拓展部分用于规定智能体的形状、大小、尺寸等仅用于render的信息，在此部分仅用于构造对应的基础动作等。
'''
class Agent:
    def __init__(self, name, length, width, movable, collided, color, init_spd, max_spd, acc, max_rng, det_rng):
        super(Agent,self).__init__()
        # TODO 完善Agent基类对应的一些基本属性，便于之后进行策略更新。
        # 设定智能体的ID，便于之后进行dict映射
        self.name = None
        # length of the agent(m)
        self.length = None
        # width of the agent(m)
        self.width = None
        # agent can move
        self.movable = False
        # agent has collided with others
        self.collided = False
        # color
        self.color = None
        # initial speed(节，1节≈1.852km/h)
        self.init_spd = None
        # maximum speed(节，1节≈1.852km/h)
        self.max_spd = None
        # acceleration(m/s^2)
        self.acc = None
        # maximun range(km)
        self.max_rng = None
        # detection range(km)
        self.det_rng = None

        # state
        self.state = AgentState()

        # action
        self.action=Action()
    
    # env step总体思路
    # 以下代码全都不需要看。
    # from math import sqrt,sin,cos
    # # 加速函数
    # def accelrate(self):
    #     self.v+=self.accelr

    # def changeAccesor(self,a):
    #     self.accelr=a

    # # 每个时间步长移动一次
    # def step(self):
    #     # 策略:先加速，再移动
    #     self.v+=self.accelr

    #     self.x+=self.v*cos(self.Angle)
    #     self.y+=self.v*sin(self.self.Angle)

    # def get_obs(self):
    #     return f"加速度:({self.accelr})；速度:({self.v});坐标(x,y):({self.x},{self.y})"
    
    # # 返回与其他智能之间的欧氏距离
    # def get_dist_with_other(self,other):
    #     return sqrt((self.x-other.x)**2+(self.x-other.x)**2)
    
    