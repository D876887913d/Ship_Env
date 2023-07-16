from pettingzoo.mpe import simple_tag_v3
from time import sleep
from utils import *

env = simple_tag_v3.env(render_mode='human')

env.reset()

action_list=[]

# print(env.action_space)

# pettingzoo获取动作空间的方法
action_dims=PzGetSpaceDimD(env.action_spaces)
obs_dims=PzGetSpaceDimC(env.observation_spaces)
print(obs_dims)
print(env.observation_space.shape)
# 返回的是总的一个状态数
print(env.state_space)

# print(list(env.observation_spaces.values()))

# for agent in env.agent_iter():
#     action = env.action_space(agent)
#     action_list.append(action)
#     print(action)
# for action_space in env.action_spaces:
#     print(action_space.n)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
    
#     if termination or truncation:
#         action = None
#     else:
#         action = env.action_space(agent).sample() # this is where you would insert your policy
        
#     env.step(action) 
#     sleep(0.01)
# env.close()