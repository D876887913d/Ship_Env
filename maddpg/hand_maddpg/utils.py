from dep import *


def onehot():
    # 具体参考
    '''
    def onehot_from_logits(logits, eps=0.01):
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        # 生成随机动作,转换成独热形式
        rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
            np.random.choice(range(logits.shape[1]), size=logits.shape[0])
        ]],
                                        requires_grad=False).to(logits.device)
        # 通过epsilon-贪婪算法来选择用哪个动作
        return torch.stack([
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ])
    '''
    dims=3
    episilon=0.5
    maxAction=[1,2,3]
    # 
    randomAction=[1,1,1]
    # 这个很妙，直接生成randlist，用于进行e-greedy算法
    for i,r in enumerate(torch.rand(dims)):
        if r>episilon:
            return maxAction[i]
        else:
            return randomAction[i]
    
class ReplayBuffer:
    def __init__(self, capacity):
        # 创建队列，存取数据
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        # 添加新的元组到经验回放池
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        # 随机采样batch size个元素
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)
    
# 离散空间获取维度
def PzGetSpaceDimD(envspaces):
    # 返回值为Discrete内部的数据
    # i.e. envspaces=env.action_spaces
    action_dim=list(envspaces.values())
    action_dim=[i.n for i in action_dim]
    return action_dim

def PzGetSpaceDimC(envspaces):
    # 返回值为上下界的元组
    # i.e. envspaces=env.action_spaces
    action_dim=list(envspaces.values())
    action_dim=[(i.low,i.high) for i in action_dim]
    return action_dim




# define the actor network
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