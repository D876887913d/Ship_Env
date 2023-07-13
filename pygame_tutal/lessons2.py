import pygame
import sys,time
from pygame.locals import *
import random

sys.path.append(".")

from Ship_ord_obj import BB1_Agent,RA1_Agent,RA2_Agent,RA3_Agent

def create_Render(displaysurface,agent_ie):
    pygame.draw.circle(displaysurface,agent_ie.color,agent_ie.state.p_pos,agent_ie.state.s_scope)

# pygame初始化
pygame.init()

# 设定坐标维度
vec = pygame.math.Vector2  # 2 for two dimensional

# config the shape of Windows
HEIGHT = 500
WIDTH = 500
ACC = 0.5
FRIC = -0.12
FPS = 60

# 图片设定帧数记录
FramePerSec = pygame.time.Clock()

# exec the configs
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ship_env")

# 设定为蓝A智能体
agent_ie=BB1_Agent()
agent_ie.state.p_pos=[255,255]
agent_ie.state.p_vel=1
agent_ie.action.aclr=0
agent_ie.action.agle=90

# 设定红A智能体
agent_ia=RA1_Agent()
agent_ia.state.p_pos=[100,100]
agent_ia.state.p_vel=1
agent_ia.action.aclr=0
agent_ia.action.agle=0

# 设置红B1智能体
agent_ia1=RA2_Agent()
agent_ia1.state.p_pos=[300,300]
agent_ia1.state.p_vel=1
agent_ia1.action.aclr=0
agent_ia1.action.agle=90

# 设置红B2
agent_ia2=RA3_Agent()
agent_ia2.state.p_pos=[250,300]
agent_ia2.state.p_vel=1
agent_ia2.action.aclr=0
agent_ia2.action.agle=0

agents=[agent_ia,agent_ia1,agent_ia2,agent_ie]

while True:	
    # 检测是否关闭游戏
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
 
    
    # 设定背景颜色为黑色
    displaysurface.fill((255,255,255))

    for i in agents:
        create_Render(displaysurface,i)
        i.update()
    

    # 时刻更新
    pygame.display.update()

    FramePerSec.tick(FPS)