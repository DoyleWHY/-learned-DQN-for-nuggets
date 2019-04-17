"""
Actor critic nugget example.

This script is the environment part of this example.
"""

from gym import spaces
import numpy as np
import time

class Nuggets(object):
    def __init__(self):
        self.action_space = ['left','right'] # available actions
        self.n_actions = len(self.action_space)
        self.n_features = 6
        self.FRESH_TIME = 0.01         #fresh time for one move
        self.action_space = spaces.Box(low=-7, high=7, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=6, shape=(1,), dtype=np.float32)
        self._build_nuggets()
        
    def _build_nuggets(self):
        env_list = ['-']*(self.n_features-1) +['T'] + ['-']*(self.n_features-1) # '-----T' our initial environment
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')

    def reset(self):
        # return observation
	    self.step_counter = 0
	    return np.array([[0]])
	    
    def render(self, S, episode, step_counter):
		# This is how environment be updated
	    env_list = ['-']*(self.n_features-1) +['T'] + ['-']*(self.n_features-1) # '--------T' our environment
	    disPositionS = int(S[0,0])
	    if S[0,0] >=6.05 or S[0,0] <=5.95:
		    env_list[disPositionS] = 'o'
		    interaction = ''.join(env_list)
		    print('\r{}'.format(interaction), end='')
		    time.sleep(self.FRESH_TIME)	
	    
		    
    def step(self, S, action):
        # This is how agent will interact with the environment 
	    disPositionS = S[0,0]
	    disPositionA = action[0,0]
	    disPositionS = disPositionS + disPositionA

	    if disPositionS >= 5.95 and disPositionS <= 6.05 :
		    reward = 20
		    done = True
	    elif disPositionS > 6.05:  
		    reward = -(6.05-disPositionS)**2
		    done = False
	    else:
		    reward = -(disPositionS - 5.95)**2
		    done = False
		    
	    if disPositionS <= 0:
		    S_ = np.array([[0]])
	    elif disPositionS >= 10:
		    S_ = np.array([[10]])
	    else:
		    S_ = np.array([[disPositionS]])
	    return S_, reward, done
        
        


