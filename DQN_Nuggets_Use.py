import tensorflow as tf
import numpy as np
import time

np.random.seed(1)
tf.set_random_seed(1)

FRESH_TIME = 1

s = tf.placeholder(tf.float32, [None,6], name='s')
sess = tf.Session()


def load_model():
	new_saver = tf.train.import_meta_graph('./model_save_dir/DQNNuggetsModel.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./model_save_dir'))
	graph = tf.get_default_graph()
	writer_test=tf.summary.FileWriter('logs',sess.graph)
	
	w1 = graph.get_tensor_by_name("target_net/l1/w1:0")
	b1 = graph.get_tensor_by_name("target_net/l1/b1:0")
	l1 = tf.nn.relu(tf.matmul(s_, w1) + b1)
	w2 = graph.get_tensor_by_name("target_net/l2/w2:0")
	b2 = graph.get_tensor_by_name("target_net/l2/b2:0")
	q_next = tf.matmul(l1, w2) + b2

	print(sess.run(q_next,feed_dict={s_: [[0,0,1,0,0,0]]}))
	print("Load Deep Q Network successfully!")

def initial_state():
    index_initial = np.random.randint(0, 5)
    s = np.array([[0, 0, 0, 0, 0, 0]])
    s[0,index_initial] = 1
    return s

def choose_action(observation):
	# forward feed the observation and get q value for every actions
	actions_value = sess.run(q_eval, feed_dict={s: observation})
	if actions_value[0,0] == actions_value[0,1]:
		action = np.random.randint(0, 2)
	else:
		action = np.argmax(actions_value)
	return action	
		
def step(S, action):
	# This is how agent will interact with the environment
	disPositionIndex = np.argwhere(S == 1)
	disPositionS = disPositionIndex[0,1]
	if action == 1 :
		disPositionS += 1
		if disPositionS == 5 :  #terminate
			done = True
		else :
			done = False
	else :  # move left
		done = False
		if disPositionS == 0 :
			S_ = S  # reach the wall
		else : 
			disPositionS -= 1 
	
	S_ = np.array([[0,0,0,0,0,0]])
	S_[0, disPositionS] = 1
	return S_, done
	
def render(S, episode, step_counter):
	# This is how environment be updated
	env_list = ['-']*(5) +['T']  # '--------T' our environment
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
		time.sleep(FRESH_TIME)	

	   
if __name__ == "__main__":
    # test Deep Q network nuggets game
    with tf.Session() as sess:
	    new_saver = tf.train.import_meta_graph('./model_save_dir/DQNNuggetsModel.meta')
	    new_saver.restore(sess, tf.train.latest_checkpoint('./model_save_dir'))
	    graph = tf.get_default_graph()
	    writer_test=tf.summary.FileWriter('logs',sess.graph)
	
	    w1 = graph.get_tensor_by_name("eval_net/l1/w1:0")
	    b1 = graph.get_tensor_by_name("eval_net/l1/b1:0")
	    l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
	    w2 = graph.get_tensor_by_name("eval_net/l2/w2:0")
	    b2 = graph.get_tensor_by_name("eval_net/l2/b2:0")
	    q_eval = tf.matmul(l1, w2) + b2

	    print(sess.run(q_eval,feed_dict={s: [[1,0,0,0,0,0]]}))
	    print("Load Deep Q Network successfully!")
	    print("Game start!")
	    
	    for episode in range(10) :
		    episode += 1
		    step_counter = 0
		    observation = initial_state()
		    #print(observation)
		    while True:
			    step_counter += 1
			    action = choose_action(observation)
			    observation, done = step(observation, action)
			    render(observation, episode, step_counter)
			    # break while loop when end of this episode
			    if done:
				    break



