from tianshou.env import PettingZooEnv       # wrapper for PettingZoo environments
from pettingzoo.classic import tictactoe_v3  # the Tic-Tac-Toe environment to be wrapped
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy, MultiAgentPolicyManager


# This board has 3 rows and 3 cols (9 places in total)
# Players place 'x' and 'o' in turn on the board
# The player who first gets 3 consecutive 'x's or 'o's wins

env = PettingZooEnv(tictactoe_v3.env(render_mode="human"))
obs = env.reset()
# env.render()                 


# agents should be wrapped into one policy,
# which is responsible for calling the acting agent correctly
# here we use two random agents
policy = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

# need to vectorize the environment for the collector
env = DummyVectorEnv([lambda: env])

# use collectors to collect a episode of trajectories
# the reward is a vector, so we need a scalar metric to monitor the training
collector = Collector(policy, env)

# you will see a long trajectory showing the board status at each timestep
result = collector.collect(n_episode=1, render=.1)