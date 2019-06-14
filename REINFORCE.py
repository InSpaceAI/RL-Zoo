import gym
import numpy as np
import tensorflow as tf

env_name = 'CartPole-v1'
state_size = 4
action_size = 2

learning_rate = 5e-3
discount_factor = 0.99
episode_num = 1000

class Model:
    def __init__(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        self.fc1 = tf.layers.dense(self.state, 256, activation=tf.nn.relu)
        self.policy = tf.layers.dense(self.fc1, action_size, activation=tf.nn.softmax)
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.action_one_hot = tf.one_hot(indices=self.action, depth=action_size)
        self.return_value = tf.placeholder(dtype=tf.float32, shape=[None])
        self.cross_entropy = tf.log(tf.reduce_sum(self.action_one_hot*self.policy, axis=1))*self.return_value
        self.loss = - tf.reduce_sum(self.cross_entropy)
        self.model_update = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

class REINFORCEAgent:
    def __init__(self):
        self.model = Model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_policy(self, state):
        policy = self.sess.run(self.model.policy, feed_dict={self.model.state: state})
        return policy

    def train_model(self, states, actions, rewards):
        return_values = self.get_return_value(rewards)
        return_values = (return_values - np.mean(return_values)) / (np.std(return_values) + 1e-6)
        _, loss = self.sess.run([self.model.model_update, self.model.loss],
                                            feed_dict = {self.model.state: states,
                                                        self.model.action: actions,
                                                        self.model.return_value: return_values})
        return loss
    
    def get_return_value(self, rewards):
        return_value = np.copy(rewards)
        for t in reversed(range(0, len(rewards)-1)):
            return_value[t] += (discount_factor * return_value[t+1])
        return return_value

if __name__ == "__main__":
    env = gym.make(env_name)
    agent = REINFORCEAgent()

    for episode in range(episode_num):
        done = False
        state = env.reset()
        states, actions, rewards = [], [], []
        while not done:
            #env.render()
            policy = agent.get_policy([state])[0]
            action = np.random.choice(action_size, 1, p=policy)[0] 
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                score = np.sum(rewards)
                loss = agent.train_model(states, actions, rewards)
                print(f"{episode + 1} Episode / Score:{score} / Loss:{loss}")
    env.close()