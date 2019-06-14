import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

env_name = 'CartPole-v1'
state_size = 4
action_size = 2

learning_rate = 8e-5
discount_factor = 0.99

epsilon_init = 1.0
epsilon_min = 0.01
epsilon_decay = 0.998

batch_size = 64
train_start_step = 1000
episode_num = 1000

class Model:
    def __init__(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        self.fc1 = tf.layers.dense(self.state, 64, activation=tf.nn.relu, kernel_initializer=tf.initializers.he_uniform(1))
        self.fc2 = tf.layers.dense(self.fc1, 64, activation=tf.nn.relu, kernel_initializer=tf.initializers.he_uniform(1))
        self.q_value = tf.layers.dense(self.fc2, action_size)
        self.action = tf.arg_max(self.q_value, 1)

        self.target_q = tf.placeholder(dtype=tf.float32, shape=[None, action_size])
        self.loss = tf.losses.mean_squared_error(self.target_q, self.q_value)
        self.model_update = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

class DQNAgent:
    def __init__(self):
        self.epsilon = epsilon_init
        self.memory = deque(maxlen=2000)
        self.model = Model()
        self.target_model = Model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.update_target = []
        for i in range(len(self.model.trainable_var)):
            self.update_target.append(self.target_model.trainable_var[i].assign(self.model.trainable_var[i]))
        self.sess.run(self.update_target)

    def get_action(self, state):
        if self.epsilon > np.random.rand():
            if self.epsilon > epsilon_min:
                self.epsilon *= epsilon_decay
            return np.random.randint(0, action_size)
        else:
            return self.sess.run(self.model.action, feed_dict={self.model.state: state})[0]
    
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        mini_batch= random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, state_size))
        next_states = np.zeros((batch_size, state_size))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
        target = self.sess.run(self.model.q_value, feed_dict={self.model.state: states})
        target_val = self.sess.run(self.target_model.q_value, feed_dict={self.target_model.state: next_states})
        for i in range(batch_size):
            target[i][actions[i]] = rewards[i] + ((1-dones[i]) * (discount_factor * np.amax(target_val[i])))
        loss, _ = self.sess.run([self.model.loss, self.model.model_update], feed_dict={self.model.state: states, self.model.target_q: target})
        return loss

if __name__ == "__main__":
    env = gym.make(env_name)
    agent = DQNAgent()
    for episode in range(episode_num):
        done = False
        score = 0
        losses = []
        state = env.reset()
        while not done:
            score += 1
            action = agent.get_action([state])
            next_state, reward, done, info = env.step(action)
            reward = reward if not done or score == 500 else -100
            agent.append_sample(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > train_start_step:
                loss = agent.train_model()
                losses.append(loss)
                if done:
                    agent.sess.run(agent.update_target)
                    print(f"{episode + 1} Episode / Score:{score} / Buffer Size:{len(agent.memory)} / Loss:{np.mean(losses)} / Epsilon:{agent.epsilon} ")

    env.close()