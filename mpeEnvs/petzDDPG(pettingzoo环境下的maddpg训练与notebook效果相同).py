from pettingzoo.mpe import simple_tag_v3
from utils import *

# 获取参数并创建环境
args = get_args()

env = simple_tag_v3.env(render_mode='human')
env.reset()

# env.agents
# ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
# n_player：总的智能体个数
# n_agents：adv智能体个数
# obs_shape：adv智能体观测空间维度列表
# action_shape：adv智能体动作空间维度列表

args.n_players = len(env.agents)
args.n_agents = args.n_players-1
args.obs_shape = []
args.action_shape=[]
for i in range(args.n_agents):  
    cur_agent=env.agents[i]
    args.obs_shape.append(env.observation_spaces[cur_agent].shape[0])
    args.action_shape.append(env.action_spaces[cur_agent].n)

args.high_action = 1
args.low_action = -1

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy
        
    env.step(action) 
env.close()
