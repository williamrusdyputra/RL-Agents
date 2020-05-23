import numpy as np
import tensorflow as tf


class ReinforceAgent:
    def __init__(self, env):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        self.gradients = []
        self.observations = []
        self.rewards = []
        self.discount = 0.99
        self.learning_rate = 1e-2
        self.network = self._build_network()

    def _build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=16, input_shape=self.state_shape, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=self.action_space, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='categorical_crossentropy')
        return model

    def store_data(self, observation, reward, action, prob):
        self.observations.append(observation)
        self.rewards.append(reward)

        encoded_action = np.zeros(self.action_space)
        encoded_action[action] = 1

        self.gradients.append(encoded_action - prob)

    def choose_action(self, observation):
        prob = self.network.predict(observation.reshape([1, -1])).flatten()
        action = np.random.choice(self.action_space, 1, p=prob)[0]
        return action, prob

    def update_network(self):
        total_reward = 0
        discounted_reward = []
        for reward in self.rewards[::-1]:
            total_reward = reward + self.discount * total_reward
            discounted_reward.append(total_reward)
        discounted_reward.reverse()
        discounted_reward = np.vstack(discounted_reward)

        mean_rewards = np.mean(discounted_reward)
        std_rewards = np.std(discounted_reward)
        discounted_reward = (discounted_reward - mean_rewards) / (std_rewards + 1e-8)

        self.gradients *= discounted_reward

        observations = np.vstack(self.observations)

        self.network.train_on_batch(observations, self.gradients)

    def reset_data(self):
        self.observations = []
        self.gradients = []
        self.rewards = []
