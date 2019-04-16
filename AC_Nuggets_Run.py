"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
nuggets game

"""

from AC_Nuggets_Env import Nuggets
from AC_Nuggets_Brain import Actor
from AC_Nuggets_Brain import Critic
import matplotlib.pyplot as plt
import tensorflow as tf

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 50
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = True  # rendering wastes time
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

def run_nuggets():
	for i_episode in range(MAX_EPISODE):
		observation = env.reset()
		t = 0
		track_r = []

		while True:
			env.render(observation, i_episode, env.step_counter)

			action = actor.choose_action(observation)

			observation_, reward, done = env.step(observation, action)

			if done: reward = -20

			track_r.append(reward)

			td_error = critic.learn(observation, reward, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
			actor.learn(observation, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

			observation = observation_
			t += 1

			if done or t >= MAX_EP_STEPS:
				ep_rs_sum = sum(track_r)

				if 'running_reward' not in globals():
					running_reward = ep_rs_sum
				else:
					running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
				if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
				print("  episode:", i_episode, "  reward:", int(running_reward), "  steps:", int(t))
				break

"""
				if i_episode == 0:
					plt.plot(vt)    # plot the episode vt
					plt.xlabel('episode steps')
					plt.ylabel('normalized state-action value')
					plt.show()
				break
				"""

	
if __name__ == "__main__":
    # nuggets game
    env = Nuggets()   
       
    sess = tf.Session()              
    actor = Actor(sess, n_features=env.n_features, n_actions=env.n_actions, lr=LR_A)
    critic = Critic(sess, n_features=env.n_features, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    run_nuggets()
