import gym
import numpy as np
import tensorflow as tf

env_name = 'CartPole-v1'
state_size = 4
action_size = 2

actor_lr = 1e-3
critic_lr = 5e-3
discount_factor = 0.99
episode_num = 1000

class Actor:
    def __init__(self):
        self.state = tf.placeholder(tf.float32, [None, state_size])
        self.fc1 = tf.layers.dense(self.state, 256, activation=tf.nn.relu)
        self.policy = tf.layers.dense(self.fc1, action_size, activation=tf.nn.softmax)
        self.action = tf.placeholder(tf.int32, [None])
        self.action_one_hot = tf.one_hot(indices=self.action, depth=action_size)
        self.advantage = tf.placeholder(tf.float32, [None])
        self.cross_entropy = tf.log(tf.reduce_sum(self.action_one_hot*self.policy, axis=1))*self.advantage
        self.loss = -tf.reduce_sum(self.cross_entropy)
        self.model_update = tf.train.AdamOptimizer(actor_lr).minimize(self.loss)

class Critic:
    def __init__(self):
        self.state = tf.placeholder(tf.float32, [None, state_size])
        self.fc1 = tf.layers.dense(self.state, 256, activation=tf.nn.relu)
        self.predict_value = tf.layers.dense(self.fc1, 1)
        self.target_value = tf.placeholder(tf.float32, [None])
        self.loss = tf.losses.mean_squared_error([self.target_value], self.predict_value)
        self.model_update = tf.train.AdamOptimizer(critic_lr).minimize(self.loss)

class A2CAgent:
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_policy(self, state):
        policy = self.sess.run(self.actor.policy, feed_dict={self.actor.state: state})
        return policy
    
    def train_model(self, state, action, reward, next_state, done):
        value = self.sess.run(self.critic.predict_value, feed_dict={self.critic.state: state})[0]
        next_value = self.sess.run(self.critic.predict_value, feed_dict={self.critic.state: next_state})[0]
        advantage = reward + ((1 - done) * discount_factor * next_value) - value
        target = reward + ((1 - done) * discount_factor * next_value)
        _, actor_loss = self.sess.run([self.actor.model_update, self.actor.loss], feed_dict={self.actor.state: state,
                                                                                            self.actor.action: action,
                                                                                            self.actor.advantage: advantage})
        _, critic_loss = self.sess.run([self.critic.model_update, self.critic.loss], feed_dict={self.critic.state: state,
                                                                                                self.critic.target_value: target})
        return actor_loss, critic_loss
    
if __name__ == "__main__":
    env = gym.make(env_name)
    agent = A2CAgent()
    for episode in range(episode_num):
        done = False
        state = env.reset()
        score = 0
        while not done:
            #env.render()
            score += 1
            policy = agent.get_policy([state])[0]
            action = np.random.choice(action_size, 1, p=policy)[0] 
            next_state, reward, done, info = env.step(action)
            actor_loss, critic_loss = agent.train_model([state], [action], reward, [next_state], done)
            state = next_state
            if done:
                print(f"{episode + 1} Episode / Score:{score} / Actor Loss:{actor_loss} / Critic Loss:{critic_loss}")

    env.close()