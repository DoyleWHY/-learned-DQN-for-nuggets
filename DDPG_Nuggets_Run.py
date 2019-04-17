"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Nuggets example.

"""

from DDPG_Nuggets_Env import Nuggets
from DDPG_Nuggets_Brain import DDPG
import numpy as np
import time 

OUTPUT_GRAPH = False
RENDER = True

MAX_EPISODES = 200
MAX_EP_STEPS = 200

def run_nuggets():
	var = 2
	t1 = time.time()
	for i in range(MAX_EPISODES):
	    s = env.reset()
	    ep_reward = 0
	    steps = 0
	    while True:
		    steps += 1
		    if RENDER:
			    env.render(s, i, env.step_counter)

		    # Add exploration noise
		    a = ddpg.choose_action(s)
		    print(a)
		    a = np.clip(np.random.normal(a, var), -1.33, 1.33)    # add randomness to action selection for exploration
		    print(a)
		    s_, r, done = env.step(s, a)
		    #print(r)
		    print(s_)

		    ddpg.store_transition(s, a, r / 10, s_)

		    if ddpg.pointer > ddpg.MEMORY_CAPACITY:
			    var *= .9995    # decay the action randomness
			    ddpg.learn()

		    s = s_
		    ep_reward += r
		    print('Cycles:',i,'  Steps:',steps)
		    if done or steps >= MAX_EP_STEPS:
			    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, ' Steps:',steps)
			    # if ep_reward > -300:RENDER = True
			    break
	print('Running time: ', time.time() - t1)
	
if __name__ == "__main__":
    # nuggets game
    env = Nuggets() 
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    
    ddpg = DDPG(a_dim, s_dim, a_bound)
       

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs_DDPG/", sess.graph)

    run_nuggets()
