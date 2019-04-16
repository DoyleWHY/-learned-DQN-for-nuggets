"""
Actor critic nugget example.

This script is the environment part of this example.
"""

import numpy as np
import time

class Nuggets(object):
    def __init__(self):
        self.action_space = ['left','right'] # available actions
        self.n_actions = len(self.action_space)
        self.n_features = 6
        self.FRESH_TIME = 0.1         #fresh time for one move
        self._build_nuggets()
        
    def _build_nuggets(self):
        env_list = ['-']*(self.n_features-1) +['T']  # '-----T' our initial environment
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')

    def reset(self):
        # return observation
	    self.step_counter = 0
	    return np.array([[1, 0, 0, 0, 0, 0]])
	    
    def render(self, S, episode, step_counter):
		# This is how environment be updated
	    env_list = ['-']*(self.n_features-1) +['T']  # '--------T' our environment
	    disPositionIndex = np.argwhere(S == 1)
	    disPositionS = disPositionIndex[0,1]
	    if disPositionS == 5:
		    interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
		    print('\r{}'.format(interaction), end='')
		    time.sleep(2)
		    print('\r           ', end='')
	    else :
		    env_list[disPositionS] = 'o'
		    interaction = ''.join(env_list)
		    print('\r{}'.format(interaction), end='')
		    time.sleep(self.FRESH_TIME)	
		    
    def step(self, S, action):
        # This is how agent will interact with the environment
	    disPositionIndex = np.argwhere(S == 1)
	    disPositionS = disPositionIndex[0,1]
	    if action == 1 :
		    disPositionS += 1
		    if disPositionS == self.n_features - 1 :  #terminate
			    reward = 1
			    done = True
		    else :
			    reward = -1
			    done = False
	    else :  # move left
		    reward = -1
		    done = False
		    if disPositionS == 0 :
		    	S_ = S  # reach the wall
		    else : 
			    disPositionS -= 1 
	    
	    S_ = np.array([[0,0,0,0,0,0]])
	    S_[0, disPositionS] = 1
	    return S_, reward, done
        
        

