from Ship_simpleEnv import SimpleEnv
from time import sleep
from Ship_scenario import Scenario

scenario = Scenario()
world = scenario.make_world()
env=SimpleEnv(scenario=scenario,world=world,max_cycles=1000)
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy
        
    env.step(action) 
    sleep(0.01)
env.close()
