import numpy as np
import cv2
import tensorflow as tf


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        self.replay_memory = []
        self.discount = 0.95
        self.epsilon = 0.1
        self.network = self._build_network()

    def _build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
            tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=self.action_space)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def store_data(self, observation, action, reward, new_observation, terminal):
        memory = [observation, action, reward, new_observation, terminal]
        self.replay_memory.append(memory)

    def pre_process_observation(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110, :]
        _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(observation, (84, 84)).astype('float32')

    def choose_action(self, observation):
        prob = self.network.predict(observation).flatten()
        greedy_action = np.argmax(prob)
        random_prob = np.random.random()
        if random_prob <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return greedy_action

    def update_network(self):
        memory = np.random.choice(self.replay_memory)
        terminal = memory[4]
        if terminal:
            action_target = memory[2]
        else:
            action_target = memory[2] + self.discount * np.max(self.network.predict(memory[3]).flatten())

        target = self.network.predict(memory[0]).flatten()
        target[memory[1]] = action_target

        self.network.train_on_batch(memory[0], target)
