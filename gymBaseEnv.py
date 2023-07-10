import gym
import math
import numpy as np

# 按照gym的方式创建一个viewer, 使用self.scale控制缩放大小
from gym.envs.classic_control import rendering

class Ball():
    def __init__(self, x: np.float32, y: np.float32, score: np.float32, way: np.int, t: np.int):
        '''
            x   初始x坐标
            y   初始y坐标
            s   初始分
            w	移动方向，弧度值
            t   球类型
        '''
        self.x = x
        self.y = y
        self.s = score
        self.w = way * 2 * math.pi / 360.0  # 角度转弧度
        self.t = t

        self.id = GenerateBallID()      # 生成球唯一id
        self.lastupdate = time.time()   # 上一次的计算时间
        self.timescale = 100            # 时间缩放，或者速度的缩放

    def update(self, way):
        '''
            更新球的状态
        '''

        # 如果是agent球，那么就改变方向
        if self.t == BALL_TYPE_SELF:
            self.w = way * 2 * math.pi / 360.0  # 角度转弧度

        speed = 1.0 / self.s    # 分数转速度大小
        now = time.time()       # 当前时间值

        self.x += math.cos(self.w) * speed * (now - self.lastupdate) * self.timescale   # 距离=速度*时间
        self.y += math.sin(self.w) * speed * (now - self.lastupdate) * self.timescale   

        self.x = CheckBound(0, VIEWPORT_W, self.x)
        self.y = CheckBound(0, VIEWPORT_H, self.y)

        self.lastupdate = now   # 更新计算时间

    def addscore(self, score: np.float32):
        self.s += score

    def state(self):
        return [self.x, self.y, self.s, self.t]




class BallEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.w=400
        self.h=400
        self.mBallNum=10
    def reset(self):
        # 管理所有球的列表， reset时先清空
        self.balls = []

        # 随机生成MAX_BALL_NUM - 1个其它球
        for i in range(self.mBallNum - 1):
            self.balls.append(self.randball(BALL_TYPE_OTHER))

        # 生成agent球
        self.selfball = self.randball(BALL_TYPE_SELF)

        # 把agent球加入管理列表
        self.balls.append(self.selfball)

        # 更新观察数据
        self.state = np.vstack([ball.state() for (_, ball) in enumerate(self.balls)])

        # 返回
        return self.state

    @staticmethod
    def randball(_t: np.int):
        # _t 为球类型（agent或其它）
        # Ball class 参数为坐标x,y, 分数score, 类型_t
        # VIEWPORT_W, VIEWPORT_H为地图宽高
        _b = Ball(np.random.rand(1)[0]*VIEWPORT_W, np.random.rand(1)[0]*VIEWPORT_H, np.random.rand(1)[0] * MAX_BALL_SCORE, np.int(np.random.rand(1)[0] * 360), _t)
        return _b
    
    def step(self, action):
        reward = 0.0	# 奖励初始值为0
        done = False	# 该局游戏是否结束

        # 首先调用ball.update方法更新球的状态
        for _, ball in enumerate(self.balls):
            ball.update(action)

        # 然后处理球之间的吞噬
        # 定一个要补充的球的类型列表，吃了多少球，就要补充多少球
        _new_ball_types = []
        # 遍历，这里就没有考虑性能问题了
        for _, A_ball in enumerate(self.balls):
            for _, B_ball in enumerate(self.balls):

                # 自己，跳过
                if A_ball.id == B_ball.id:
                    continue

                # 先计算球A的半径
                # 我们使用球的分数作为球的面积
                A_radius = math.sqrt(A_ball.s / math.pi)

                # 计算球AB之间在x\y轴上的距离
                AB_x = math.fabs(A_ball.x - B_ball.x)
                AB_y = math.fabs(A_ball.y - B_ball.y)

                # 如果AB之间在x\y轴上的距离 大于 A的半径，那么B一定在A外
                if AB_x > A_radius or AB_y > A_radius:
                    continue

                # 计算距离
                if AB_x*AB_x + AB_y*AB_y > A_radius*A_radius:
                    continue

                # 如果agent球被吃掉，游戏结束
                if B_ball.t == BALL_TYPE_SELF:
                    done = True

                # A吃掉B A加上B的分数
                A_ball.addscore(B_ball.s)

                # 计算奖励
                if A_ball.t == BALL_TYPE_SELF:
                    reward += B_ball.s

                # 把B从列表中删除，并记录要增加一个B类型的球
                _new_ball_types.append(B_ball.t)
                self.balls.remove(B_ball)

        # 补充球
        for _, val in enumerate(_new_ball_types):
            self.balls.append(self.randball(np.int(val)))

        # 填充观察数据
        self.state = np.vstack([ball.state() for (_, ball) in enumerate(self.balls)])

        # 返回
        return self.state, reward, done, {}

    def render(self, mode='human'):
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.w * self.scale, self.h * self.scale)

        # 渲染所有的球
        for item in self.state:
            # 从状态中获取坐标、分数、类型
            _x, _y, _s, _t = item[0] * self.scale, item[1] * self.scale, item[2], item[3]

            # transform用于控制物体位置、缩放等
            transform = rendering.Transform()
            transform.set_translation(_x, _y)

            # 添加一个⚪，来表示球
            # 中心点: (x, y)
            # 半径: sqrt(score/pi)
            # 颜色: 其它球为蓝色、agent球为红/紫色
            self.viewer.draw_circle(math.sqrt(_s / math.pi) * self.scale, 30, color=(_t, 0, 1)).add_attr(transform)

        # 然后直接渲染（没有考虑性能）
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
if __name__ == '__main__':
    env = BallEnv()
    
    while True:
        env.step(150)
        env.render()
