import tensorflow as tf
import numpy as np
import gym
import copy

env_name = 'CartPole-v1'
state_size = 4
action_size = 1

_lambda = 0.95
ppo_eps = 0.1
epoch = 3
learning_rate = 5e-3
gamma = 0.99

episode_num = 1000
n_step = 20

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5*(((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_gaes(rewards, dones, values, next_values, gamma, _lambda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * _lambda * gaes[t + 1]
    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target

class Actor:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.action = tf.placeholder(tf.float32, [None, action_size])

            self.fc1 = tf.layers.dense(self.state, 256, tf.nn.relu)
            self.mu = tf.layers.dense(self.fc1, action_size)
            self.log_std = tf.get_variable("log_std", initializer=-0.5*np.ones(action_size, np.float32))
            self.std = tf.exp(self.log_std)
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu))*self.std
            self.logp = gaussian_likelihood(self.action, self.mu, self.log_std)
            self.logp_pi = gaussian_likelihood(self.pi, self.mu, self.log_std)

class Critic:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 256, tf.nn.relu)
            self.value = tf.layers.dense(self.fc1, 1)
            self.v = tf.squeeze(self.value, axis=1)

class PPOAgent:
    def __init__(self):
        self.actor = Actor("Actor")
        self.critic = Critic("Critic")

        self.adv = tf.placeholder(tf.float32, [None])
        self.ret = tf.placeholder(tf.float32, [None])
        self.logp_old = tf.placeholder(tf.float32, [None])

        self.ratio = tf.exp(self.actor.logp - self.logp_old)
        self.min_adv = tf.where(self.adv > 0, (1.0+ppo_eps)*self.adv, (1.0-ppo_eps)*self.adv)
        self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio*self.adv, self.min_adv))
        self.v_loss = tf.reduce_mean((self.ret-self.critic.v)**2)

        self.train_actor = tf.train.AdamOptimizer(learning_rate).minimize(self.pi_loss)
        self.train_critic = tf.train.AdamOptimizer(learning_rate).minimize(self.v_loss)

        self.approx_kl = tf.reduce_mean(self.logp_old - self.actor.logp)
        self.approx_ent = tf.reduce_mean(-self.actor.logp)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update(self, state, action, target, adv, logp_old):
        v_loss, kl, ent = 0, 0, 0
        for i in range(epoch):
            _, _, sub_v_loss, approx_kl, approx_ent = \
                 self.sess.run([self.train_actor, self.train_critic, self.v_loss, self.approx_kl, self.approx_ent],
                                feed_dict={self.actor.state: state, self.critic.state: state, self.actor.action: action,
                                             self.ret: target, self.adv: adv, self.logp_old: logp_old})
            v_loss += sub_v_loss
            kl += approx_kl
            ent += approx_ent
        return v_loss, kl, ent

    def get_action(self, state):
        action, v, logp_pi = self.sess.run([self.actor.pi, self.critic.v, self.actor.logp_pi],
                                        feed_dict={self.actor.state: state, self.critic.state: state})
        return action, v, logp_pi

if __name__ == "__main__":
    env = gym.make(env_name)
    agent = PPOAgent()
    episode = 0
    score = 0
    state = env.reset()
    while episode < episode_num:
        for t in range(n_step):
            #env.render()
            action, v_t, logp_t = agent.get_action([state])
            action, v_t, logp_t  =  action[0], v_t[0], logp_t[0]
            discrete_action = 0 if action < 0 else 1
            next_state, reward, done, info = env.step(discrete_action)
            if t == 0:
                states, actions, values, logp_ts, dones, rewards = [state], [action], [v_t], [logp_t], [done], [reward]
            else:
                states = np.r_[states, [state]]
                actions = np.r_[actions, [action]]
                values = np.r_[values, v_t]
                logp_ts = np.r_[logp_ts, logp_t]
                dones = np.r_[dones, done]
                rewards = np.r_[rewards, reward]
            state = next_state
            score += 1
            if done:
                print(f"{episode + 1} Episode / Score:{score}")
                score = 0
                state = env.reset()
                episode += 1
        
        v_t = agent.get_action([state])[1][0]
        values = np.r_[values, v_t]
        next_values = np.copy(values[1:])
        values = values[:-1]
        adv, target = get_gaes(rewards, dones, values, next_values, gamma, _lambda, True)
        agent.update(states, actions, target, adv, logp_ts)
    env.close()