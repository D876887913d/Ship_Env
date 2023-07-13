import math

class AgentState:  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # xy坐标值/速度
        # self.p_pos=(x,y) list
        self.p_pos=None
        self.p_vel=0
        # 检测范围,由于会受到指定的红A2影响，设为状态
        self.s_scope=0

class Action:  # action of the agent
    def __init__(self):
        # 角度/加速度        
        self.agle=0
        self.aclr=0

'''
:breif 这个函数对应的是智能体的基本属性，例如坐标、速度、方向等，用于之后进行进一步的环境构建的时候进行配置各智能体。
:expand 拓展部分用于规定智能体的形状、大小、尺寸等仅用于render的信息，在此部分仅用于构造对应的基础动作等。
'''
class Agent:
    def __init__(self):
        super(Agent,self).__init__()
        # TODO 完善Agent基类对应的一些基本属性，便于之后进行策略更新。
        # 设定智能体的ID，便于之后进行dict映射
        self.name=None

        # state
        self.state = AgentState()

        # action
        self.action=Action()
    
    def update(self):
        self.state.p_vel+=self.action.aclr
        self.state.p_pos[0]+=self.state.p_vel*math.cos(math.radians(self.action.agle))
        self.state.p_pos[1]+=self.state.p_vel*math.sin(math.radians(self.action.agle))

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
    
class BB1_Agent(Agent):
    def __init__(self):
        super().__init__()
        self.color=(117, 162, 228)
        self.state.s_scope=25

class RA1_Agent(Agent):
    def __init__(self):
        super().__init__()
        self.color=(255, 184, 184)
        self.state.s_scope=112.5

class RA2_Agent(Agent):
    def __init__(self):
        super().__init__()
        self.color=(247, 189, 198)
        self.state.s_scope=50

class RA3_Agent(Agent):
    def __init__(self):
        super().__init__()
        self.color=(250, 210, 200)
        self.state.s_scope=50