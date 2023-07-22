import threading
import os
from tqdm import tqdm
import numpy as np
import inspect
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 配置参数
def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    # Fix Bug
    # 如果是在colab需要解注下面这个参数
    # parser.add_argument('-f')

    # 如果在vscode需要解注下列参数
    parser.add_argument("--ip")
    parser.add_argument("--stdin")
    parser.add_argument("--control")
    parser.add_argument("--hb")
    parser.add_argument("--Session.signature_scheme")
    parser.add_argument("--Session.key")
    parser.add_argument("--shell")
    parser.add_argument("--transport")
    parser.add_argument("--iopub")
    parser.add_argument("--f")
    args = parser.parse_args()

    return args

# 经验回放池
class Buffer:
    # 初始化经验回放池，初始化尺寸、buffer内部空间初始化、线程锁初始化
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    # 存放一个经验，包括观测值o_i，动作值u_i,回报值r_i,下一个观测值o_next_i,对于每个可以学习的智能体都需要存入这四个元素
    # 存入的时候四个一同存入
    # 对应关系举例：buffer[o_1][idx]对应智能体 1
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]

    # 采样batch_size个经验
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    # 存入inc个新的元素，并返回inc个元素的下表
    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        # 如果当前的尺寸+inc<=buffer的最大尺寸，那么idx表示的是cuurent_size到cuurent_size+inc的索引值array
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)

        # 如果当前的尺寸+inc>buffer，而且当前的尺寸小于最大尺寸，那么取出来current_size之后的，然后从0~current_size之间取溢出的数值个随机数
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        
        # 如果说原本的current_size就满了，那么直接从0~size之间取inc个随机数
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

# input:  agent_i_obs_dim
# output: agent_i_action_dim
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

# input:  all_obs_dim+all_action_dim
# output: 1  (Q-value)
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

# 一代maddpg网络
class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # 创建基本的ac网络
        self.actor_network = Actor(args, agent_id).to(device)
        self.critic_network = Critic(args).to(device)

        # 构建目标网络
        self.actor_target_network = Actor(args, agent_id).to(device)
        self.critic_target_network = Critic(args).to(device)

        # 初始化目标网络参数
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # 网络优化器
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # 自动为每个智能体创建一个存储模型参数的文件夹
        # 由于os库的一些自身限制，无法直接建立多重文件夹，因此需要一级一级的构建文件夹
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # # 如果有初始模型的话，加载初始模型参数
        # if os.path.exists(self.model_path + '/actor_params.pkl'):
        #     self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
        #     self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
        #     print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
        #                                                                   self.model_path + '/actor_params.pkl'))
        #     print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
        #                                                                    self.model_path + '/critic_params.pkl'))
        # print(self.model_path )
    # 软更新目标网络参数
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # 网络训练
    def train(self, transitions, other_agents):
        # 将每一个经验都转化为tensor的形式
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(device)

        # 将当前智能体的batchsize个reward保存
        r = transitions['r_%d' % self.agent_id]

        # 用来装每个agent经验中的各项
        o, u, o_next = [], [], []  
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # 按照智能体的顺序将动作添加到动作列表中，均采用的对应的智能体的策略网络
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:                    
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            
            # 将下一个时间步的状态与预测的动作送入critic目标网络中
            q_next = self.critic_target_network(o_next, u_next).detach()

            # 计算目标的q值，具体公式为:r(奖励值)+gamma*q_next(折扣因子乘以下一个时刻的q值)
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # 利用当前的状态与动作计算q值，将目标网络的q值与实际网络的q值均方误差
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()

        # 测试loss值，判断loss是不是下降了
        # print(critic_loss,actor_loss)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()

        # 项目调试，暂不保存
        # if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
        #     self.save_model(self.train_step)
        # self.train_step += 1

    # 保存模型,传入参数为模型的序号
    def save_model(self, num):
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')

    # 将模型的参数加载到当前模型中
    def load_model(self):
        # 如果有初始模型的话，加载初始模型参数
        if os.path.exists(self.model_path + '/bst_actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/bst_actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/bst_critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/bst_actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/bst_critic_params.pkl'))
        # if os.path.exists(path + '/actor_params.pkl'):
        #     self.actor_network.load_state_dict(torch.load(path + '/bst_actor_params.pkl'))
        #     self.critic_network.load_state_dict(torch.load(path + '/bst_critic_params.pkl'))
        #     print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
        #                                                                   path + '/bst_actor_params.pkl'))
        #     print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
        #                                                                    path + '/bst_critic_params.pkl'))

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    # 用于实现噪声的添加以及epsilon-greedy算法的实现，并在最后把对应的动作获取
    def select_action(self, o, noise_rate, epsilon):        
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            pi = self.policy.actor_network(inputs).squeeze(0)
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    # 进行智能体网络的训练
    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

# 一代验证代码
def evaluate(args,env,agents):
    evaluate_episodes=10
    evaluate_episodes_len=100

    # 回报列表，用于进行reward均值图像的绘制
    return_list=[]
    for episode in range(evaluate_episodes):
        # reset the environment
        s = env.reset()
        reward=[0 for i in range(len(agents))]
        for i in range(evaluate_episodes_len):
            # env.render()
            # 单步步长移动对应的代码
            u=[]
            actions=[]

            # 为每个智能体确定动作
            with torch.no_grad():
                for agent_id,agent in enumerate(agents):
                    action=agent.select_action(s[agent_id],args.noise_rate,args.epsilon)
                    actions.append(action)

            # 为每个非智能体确定动作
            for i in range(args.n_agents,args.n_players):
                # 非智能体仅通过随机移动改变状态
                actions.append([0,np.random.rand()*2-1,0,np.random.rand()*2-1,0])

            # 获取环境反馈
            s_next,r,_,_=env.step(actions)
            
            # 状态更新
            s=s_next

            # 第0个adv的奖励总和
            # 每轮的reward
            reward=[r[i]+reward[i] for i in range(len(reward))]
    
        return_list.append(reward)

    return_list=np.array(return_list)

    for i in range(len(return_list[0])):
        print(sum(return_list[:,i])/evaluate_episodes)

# 二代验证代码
def run_evaluate(env,agents,eval_episodes,max_step_per_episode):
    '''
    params:  env表示环境
    params:  agents为用于决策的智能体的列表
    params:  eval_episodes为检验的轮数
    params:  max_step_per_episode为每次环境运行的最大步长
    '''
    eval_episode_rewards=[]
    eval_episode_steps=[]
    while len(eval_episode_rewards)<eval_episodes:
        obs_n=env.reset()
        done=False
        total_reward=0
        steps=0
        while not done and steps<max_step_per_episode:
            steps+=1
            action_n=[
                agent.predict(obs) for agent,obs in zip(agents,obs_n)
            ]
            obs_n,reward_n,done_n,_=env.step(action_n)
            done=all(done_n)
            total_reward+=sum(reward_n)
    eval_episode_rewards.append(total_reward)
    eval_episode_steps.append(steps)
    return eval_episode_rewards,eval_episode_steps

# 二代训练代码
def run_episode(env, agents,max_step_per_episode):
    '''
    breif:   进行一轮训练
    params:  env表示环境
    params:  agents为用于决策的智能体的列表
    params:  max_step_per_episode为每次环境运行的最大步长
    '''
    obs_n = env.reset()
    done = False
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while not done and steps < max_step_per_episode:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)

        # store experience
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        # compute reward of every agent
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # learn policy
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)

    return total_reward, agents_reward, steps


# def AgoodTrainExample():
#     # 这是一个伪代码
#     # 创建一个经验缓冲区，用于经验回放
#     # 一个好用的训练代码~经过自己修改的，可以直接迁移到那个simpletag中，之后可以用到speaker上
#     buffer=Buffer(args)

#     train_episodes=2000000
#     train_episodes_len=1000

#     return_list=[]
#     for episode in range(train_episodes//train_episodes_len):
#         episode_return = [0 for i in range(args.n_players)]
#         with tqdm(total=train_episodes_len, desc='Iteration %d' % episode) as pbar:
#             for i_episode in range(train_episodes_len):
#                 s = env.reset()
#                 done = False
#                 while not done:
#                     # 单步步长移动对应的代码
#                     u=[]
#                     actions=[]

#                     # 为每个智能体确定动作
#                     with torch.no_grad():
#                         for agent_id,agent in enumerate(agents):
#                             action=agent.select_action(s[agent_id],args.noise_rate,args.epsilon)
#                             u.append(action)
#                             actions.append(action)

#                     # 为每个非智能体确定动作
#                     for i in range(args.n_agents,args.n_players):
#                         # 非智能体仅通过随机移动改变状态
#                         actions.append([0,np.random.rand()*2-1,0,np.random.rand()*2-1,0])

#                     # 获取环境反馈
#                     s_next,r,done,info=env.step(actions)

#                     # 存放至缓冲区
#                     buffer.store_episode(s[:args.n_agents],u,r[:args.n_agents],s_next[:args.n_agents])

#                     # 状态更新
#                     s=s_next

#                     episode_return =[r[i]+episode_return[i] for i in range(args.n_players)]
#                     # 当buffer数据的数量超过一定值后,才进行Q网络训练
#                     if buffer.current_size>args.batch_size:
#                         transitions=buffer.sample(args.batch_size)
#                         for agent in agents:
#                             other_agents=agents.copy()
#                             other_agents.remove(agent)
#                             # 送入网络以及训练等众多事宜都是从这个函数内部进行的
#                             agent.learn(transitions,other_agents)

#                 return_list.append(episode_return)
#                 if (i_episode + 1) % 10 == 0:
#                     pbar.set_postfix({
#                         'adv_0 reward':
#                         '%.3f' % episode_return[0],
#                         'agent_0 reward':
#                         '%.3f' % episode_return[3]
#                     })
#                 pbar.update(1)