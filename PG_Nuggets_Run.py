"""
Policy Gradient, Reinforcement Learning.

The nuggets example

"""

from PG_Nuggets_Env import Nuggets
from PG_Nuggets_Brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time


def run_nuggets():
	for i_episode in range(10):

		observation = env.reset()

		while True:
			env.render(observation, i_episode, env.step_counter)

			action = RL.choose_action(observation)

			observation_, reward, done = env.step(observation, action)

			RL.store_transition(observation, action, reward)

			if done:
				ep_rs_sum = sum(RL.ep_rs)

				if 'running_reward' not in globals():
					running_reward = ep_rs_sum
				else:
					running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
				if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
				print("episode:", i_episode, "  reward:", int(running_reward))

				vt = RL.learn()

				if i_episode == 0:
					plt.plot(vt)    # plot the episode vt
					plt.xlabel('episode steps')
					plt.ylabel('normalized state-action value')
					plt.show()
				break

			observation = observation_

	
if __name__ == "__main__":
    # nuggets game
    env = Nuggets()                    
    RL = PolicyGradient(n_actions=env.n_actions,
					    n_features=env.n_features,
					    learning_rate=0.02,
					    reward_decay=0.99,
					    # output_graph=True,
					    )
    run_nuggets()
    
    
