import sys
sys.path.append(".")

from maddpg.utils import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import matplotlib.pyplot as plt

def make_env(args):
    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # env = MultiAgentEnv(world)
    args.n_players = env.n  # 包含敌人的所有玩家个数
    args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.high_action = 1
    args.low_action = -1
    return env, args

# 获取参数并创建环境
args = get_args()
env, args = make_env(args)
print("Env创建完毕!")

# 智能体更新
agents = []
for i in range(args.n_agents):
    agent = Agent(i, args)
    agents.append(agent)
    agent.policy.load_model()
print("agents创建完毕!")

# 总的训练数，对应的是整个回合的个数
num_episodes=2000000

# 回报列表，用于进行reward均值图像的绘制
return_list=[]

# 创建一个经验缓冲区，用于经验回放
buffer=Buffer(args)

# 训练开始~
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = [0 for i in range(args.n_players)]
            s = env.reset()
            done = False
            while not done:
                # 单步步长移动对应的代码
                u=[]
                actions=[]

                # 为每个智能体确定动作
                with torch.no_grad():
                    for agent_id,agent in enumerate(agents):
                        action=agent.select_action(s[agent_id],args.noise_rate,args.epsilon)
                        u.append(action)
                        actions.append(action)

                # 为每个非智能体确定动作
                for i in range(args.n_agents,args.n_players):
                    # 非智能体仅通过随机移动改变状态
                    actions.append([0,np.random.rand()*2-1,0,np.random.rand()*2-1,0])

                # 获取环境反馈
                s_next,r,done,info=env.step(actions)
                
                # 存放至缓冲区
                buffer.store_episode(s[:args.n_agents],u,r[:args.n_agents],s_next[:args.n_agents])

                # 状态更新
                s=s_next

                episode_return =[r[i]+episode_return[i] for i in range(args.n_players)]
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if buffer.current_size>args.batch_size:
                    transitions=buffer.sample(args.batch_size)
                    for agent in agents:
                        other_agents=agents.copy()
                        other_agents.remove(agent)
                        # 送入网络以及训练等众多事宜都是从这个函数内部进行的
                        agent.learn(transitions,other_agents)

            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'adv_0 reward':
                    '%.3f' % np.mean(np.array(return_list)[-10:,0]),
                    'agent_0 reward':
                    '%.3f' % np.mean(np.array(return_list)[-10:,3])
                })
            pbar.update(1)