import numpy as np
import tensorflow as tf
import gym
from collections import deque
import random

env_name = 'CartPole-v1'
state_size = 4
action_size = 1
batch_size = 32

mem_maxlen = 50000

mu = 5e-4
theta = 0.1
sigma = 0.1
epsilon_init = 1.0
epsilon_decay = 0.9999

tau = 5e-3
actor_lr = 1e-3
critic_lr = 5e-3
discount_factor = 0.99
train_start_step = 2000
episode_num = 1000

class OU_noise:
    def __init__(self):
        self.reset()
    def reset(self):
        self.X = np.ones(action_size) * mu
    def sample(self):
        dx = theta * (mu - self.X)
        dx += sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X

class Actor:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 256, activation=tf.nn.relu)
            self.action = tf.layers.dense(self.fc1, action_size, activation=tf.nn.sigmoid)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class Critic:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.action = tf.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.state, self.action],axis=-1)
            self.fc1 = tf.layers.dense(self.concat, 256, activation=tf.nn.relu)
            self.predict_q = tf.layers.dense(self.fc1, 1, activation=None)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class DDPGAgent:
    def __init__(self):
        self.actor = Actor("actor")
        self.critic = Critic("critic")
        self.target_actor = Actor("target_actor")
        self.target_critic = Critic("target_critic")
        self.epsilon = epsilon_init
        self.target_q = tf.placeholder(tf.float32, [None, 1])
        critic_loss = tf.losses.mean_squared_error(self.target_q, self.critic.predict_q)
        with tf.control_dependencies(self.critic.trainable_var):
            self.train_critic = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)

        action_grad = tf.clip_by_value(tf.gradients(tf.squeeze(self.critic.predict_q), self.critic.action),-10,10)
        policy_grad = tf.gradients(self.actor.action, self.actor.trainable_var, action_grad)
        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads/batch_size
        with tf.control_dependencies(self.actor.trainable_var):
            self.train_actor = tf.train.AdamOptimizer(actor_lr).apply_gradients(zip(policy_grad, self.actor.trainable_var))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)

        self.soft_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            self.soft_update_target.append(self.target_actor.trainable_var[idx].assign(
                ((1 - tau) * self.target_actor.trainable_var[idx].value()) + (tau * self.actor.trainable_var[idx].value())))
        for idx in range(len(self.critic.trainable_var)):
            self.soft_update_target.append(self.target_critic.trainable_var[idx].assign(
                ((1 - tau) * self.target_critic.trainable_var[idx].value()) + (tau * self.critic.trainable_var[idx].value())))

        init_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            init_update_target.append(self.target_actor.trainable_var[idx].assign(self.actor.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target.append(self.target_critic.trainable_var[idx].assign(self.critic.trainable_var[idx]))
        self.sess.run(init_update_target)

    def train_model(self):
        self.epsilon *= epsilon_decay
        mini_batch = random.sample(self.memory, batch_size)
        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_actions = self.sess.run(self.target_actor.action,
                                            feed_dict={self.target_actor.state: next_states})
        target_critic_predict_qs = self.sess.run(self.target_critic.predict_q,
                                                feed_dict={self.target_critic.state: next_states, self.target_critic.action: target_actor_actions})
        target_qs = np.asarray([reward + discount_factor * (1 - done) * target_critic_predict_q
                                for reward, target_critic_predict_q, done in zip(rewards, target_critic_predict_qs, dones)])

        self.sess.run(self.train_critic, feed_dict={self.critic.state: states, self.critic.action: actions, self.target_q: target_qs})
        actions_for_train = self.sess.run(self.actor.action, feed_dict={self.actor.state: states})
        self.sess.run(self.train_actor, feed_dict={self.actor.state: states, self.critic.state: states, self.critic.action: actions_for_train})
        self.sess.run(self.soft_update_target)

    def get_action(self, state):
        action = self.sess.run(self.actor.action, feed_dict={self.actor.state: state})[0]
        noise = self.OU.sample()
        return action + (self.epsilon*noise)
    
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

if __name__ == "__main__":
    env = gym.make(env_name)
    agent = DDPGAgent()

    for episode in range(episode_num):
        done = False
        state = env.reset()
        score = 0
        while not done:
            score += 1
            #env.render()
            action = agent.get_action([state])
            discrete_action = 0 if action < 0.5 else 1
            next_state, reward, done, info = env.step(discrete_action)
            agent.append_sample(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > train_start_step:
                agent.train_model()
            if done:
                print(f"{episode + 1} Episode / Score:{score} / Buffer Size:{len(agent.memory)}")

    env.close()