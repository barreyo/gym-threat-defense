
"""Policy Gradient implementation to solve threat defense gym problem."""

import itertools

import numpy as np
import tensorflow as tf

import gym
import gym_threat_defense  # noqa


# ONLY FOR RESULT PLOTTING
import csv


class DPGAgent(object):
    """Policy Gradient agent for solving the threat defense environment."""

    def __init__(self, env, session, num_episodes=1000, batch_size=100):
        """
        Setups the agent.

        Arguments:
        env -- OpenAI gym environment.
        session -- TensorFlow session object
        num_episodes -- Total number of episodes
        batch_size -- Number of batches to run
        """
        self.env = env
        self.session = session
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.learning_rate = 1e-2

        self.inp = tf.placeholder(
            tf.float32,
            shape=[None, 12])

        # Following the quidelines in this post:
        # https://stats.stackexchange.com/questions/181/how-to-choose-the- \
        # number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        hidden = tf.contrib.layers.fully_connected(
            inputs=self.inp,
            num_outputs=30,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.glorot_uniform_initializer())

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden,
            num_outputs=self.env.action_space.n,
            activation_fn=None)

        self.sample_action = tf.reshape(tf.multinomial(logits, 1), [])

        log_prob = tf.log(tf.nn.softmax(logits))

        self.actions = tf.placeholder(tf.int32)
        self.advantages = tf.placeholder(tf.float32)

        idxs = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + \
            self.actions
        action_probs = tf.gather(tf.reshape(log_prob, [-1]), idxs)
        loss_func = -tf.reduce_sum(tf.multiply(action_probs, self.advantages))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.minimize(loss_func)

    def get_action(self, observation):
        """
        Get sampled action from observation.

        Arguments:
        observation -- A single observation.

        Returns:
        A single sampled action based on the observation.
        """
        return self.session.run(self.sample_action,
                                feed_dict={self.inp: [observation]})

    def update(self, observations, actions, advantages):
        """
        Run an update of the neural network.

        Arguments:
        observations -- Array of observations
        actions -- Array of actions.
        advantages -- Array of advantages.
        """
        feed_dict = {self.inp: observations, self.actions: actions,
                     self.advantages: advantages}
        self.session.run(self.train, feed_dict=feed_dict)

    def _get_index_in_matrix(self, env, observation):
        for i in range(env.all_states.shape[0]):
            if np.array_equal(observation, env.all_states[i]):
                return i

    def run(self, render=True):
        """
        Run agent in environment.

        Arguments:
        render -- True if you want the environment to be render to a window,
            false if running as headless.
        """
        # Initialize the TensorFlow graph
        self.session.run(tf.global_variables_initializer())

        all_averages = []
        stds = []
        all_timesteps = []
        all_time_averages = []
        time_stds = []

        for n in xrange(self.num_episodes):

            print 'episode %s' % n

            # Keep track of our rewards(advantages), actions and observations
            # so we can feed these into the neural net
            advantages, actions, observations = [], [], []
            total_rewards = []

            for i in xrange(self.batch_size):

                obs = self.env.reset()

                total_reward = 0
                timesteps = 0
                batch_advantages = []

                for k in itertools.count():

                    timesteps += 1

                    if render:
                        self.env.render()

                    observations.append(obs)

                    # Sample an action
                    action = self.get_action(obs)
                    obs, reward, done, _ = self.env.step(action)
                    actions.append(action)
                    total_reward += reward
                    batch_advantages.append(reward)

                    if done:
                        break

                total_rewards.append(total_reward)
                all_timesteps.append(timesteps)

                advantages.extend([total_reward + timesteps] * (timesteps))

            avg_b_reward = np.mean(total_rewards)
            print 'avg rew: %s' % avg_b_reward
            print 'avg time: %s' % np.mean(all_timesteps)
            all_averages.append(avg_b_reward)
            stds.append(np.std(total_rewards))

            all_time_averages.append(np.mean(all_timesteps))
            time_stds.append(np.std(all_timesteps))

            # advantages = map(lambda x: x, advantages)
            # Normalize all rewards
            # avoid div by zero by adding some small padding
            advantages = (advantages - np.mean(advantages)) / \
                (np.std(advantages) + 1e-10)
            self.update(observations, actions, advantages)

        with open('pg_res.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            episode_numbers = ['E'] + range(1, self.num_episodes + 1)
            writer.writerows(zip(episode_numbers, ['A'] + all_averages))

        with open('pg_res_std_high.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            episode_numbers = ['E'] + range(1, self.num_episodes + 1)
            stds_up = map(lambda x: x[0] + x[1], zip(all_averages, stds))

            writer.writerows(zip(episode_numbers, ['V'] + stds_up))

        with open('pg_res_std_low.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            episode_numbers = ['E'] + range(1, self.num_episodes + 1)
            stds_up = map(lambda x: x[0] - x[1], zip(all_averages, stds))

            writer.writerows(zip(episode_numbers, ['V'] + stds_up))

        with open('pg_time.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            episode_numbers = ['E'] + range(1, self.num_episodes + 1)
            writer.writerows(zip(episode_numbers, ['T'] + all_time_averages))

        with open('pg_time_std_high.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            episode_numbers = ['E'] + range(1, self.num_episodes + 1)
            stds_up = map(lambda x: x[0] + x[1], zip(all_time_averages,
                                                     time_stds))

            writer.writerows(zip(episode_numbers, ['V'] + stds_up))

        with open('pg_time_std_low.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            episode_numbers = ['E'] + range(1, self.num_episodes + 1)
            stds_up = map(lambda x: x[0] - x[1], zip(all_time_averages,
                                                     time_stds))

            writer.writerows(zip(episode_numbers, ['V'] + stds_up))


if __name__ == '__main__':
    env = gym.make("threat-defense-v0")

    sess = tf.InteractiveSession()
    pg_agent = DPGAgent(env, sess)

    pg_agent.run(render=False)
