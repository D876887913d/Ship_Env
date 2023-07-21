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

        # 如果有初始模型的话，加载初始模型参数
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))
        print(self.model_path )
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

    # 保存模型
    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
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

