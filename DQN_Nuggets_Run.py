from DQN_Nuggets_Env import Nuggets
from DQN_Nuggets_Brain import DeepQNetwork

def run_nuggets():
	step = 0
	for episode in range(300):
        # initial observation
		observation = env.reset()
		#print(observation)

		while True:
            # fresh env
			env.render(observation, episode, env.step_counter)

            # RL choose action based on observation
			action = RL.choose_action(observation)

            # RL take action and get next observation and reward
			observation_, reward, done = env.step(observation, action)

			RL.store_transition(observation, action, reward, observation_)

			if (step > 200) and (step % 5 == 0):
				RL.learn()

            # swap observation
			observation = observation_

            # break while loop when end of this episode
			if done:
				break
			step += 1

    # end of game
	RL.test_func()
	print('game over')
	
	
if __name__ == "__main__":
    # nuggets game
    env = Nuggets()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True,
                      save_model=True
                      )
    run_nuggets()
    RL.plot_cost()
