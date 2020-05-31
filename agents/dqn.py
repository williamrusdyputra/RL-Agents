import cv2
import numpy as np
import random
import tensorflow as tf

"""
    You will notice that I use uint8 for observation and then convert them into float32 when using them.
    This is necessary because RAM usage can get very large to store frames from the game and uint8
    is the smallest datatype to store the frames.
    Numpy asarray is also used because np.array() copy the object and making more memory use whereas 
    np.asarray does not unless it is necessary 
"""


class DQNAgent:
    def __init__(self, env):
        self.path = ['./weights_dqn/network/network', './weights_dqn/target_network/target_network']
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        self.replay_memory = []
        self.discount = 0.95
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.exploration_steps = 1e6
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / self.exploration_steps
        self.batch_size = 32
        self.max_memory = 5e5
        self.c_iteration = 1e4
        self.replay_start_size = 1e6
        self.update_iteration = 0
        self.network = self._build_network('main')
        self.target_network = self._build_network('target')

    def _build_network(self, model_type):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
            tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=self.action_space)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

        if model_type == 'main':
            model.load_weights(self.path[0])
        else:
            model.load_weights(self.path[1])

        return model

    def store_data(self, observation, action, reward, new_observation, terminal):
        memory = [observation, action, reward, new_observation, terminal]
        if len(self.replay_memory) > self.max_memory:
            self.replay_memory.pop(0)
        self.replay_memory.append(memory)

    def pre_process_observation(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110, :]
        observation = np.reshape(observation, (84, 84)).astype('uint8')
        return observation / 255.0

    def choose_action(self, observation):
        observation = observation.astype('float32')
        prob = self.network.predict(observation).flatten()
        greedy_action = np.argmax(prob)
        random_prob = np.random.random()
        if random_prob <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return greedy_action

    def update_epsilon(self):
        if self.epsilon <= self.min_epsilon:
            return
        self.epsilon -= self.epsilon_decay

    def update_network(self, steps):
        if steps < self.replay_start_size:
            return

        self.update_iteration += 1
        memories = random.sample(self.replay_memory, self.batch_size)

        memory_observations = []
        memory_targets = []

        for memory in memories:
            terminal = memory[4]
            if terminal:
                action_target = memory[2]
            else:
                memory[3] = np.asarray(memory[3]).astype('float32')
                action_target = memory[2] + self.discount * np.max(self.target_network.predict(memory[3]).flatten())

            memory[0] = np.asarray(memory[0]).astype('float32')
            encoded_target = np.zeros(self.action_space)
            encoded_target[memory[1]] = action_target

            memory_observations.append(memory[0])
            memory_targets.append(encoded_target)

        memory_observations = np.asarray(memory_observations).astype('float32')
        memory_targets = np.asarray(memory_targets).astype('float32')

        # shape of memory observations is (32, 1, 84, 84, 4). We have to remove the 2nd dimension so that we get shape
        # (32, 84, 84, 4) which we can input to the network
        memory_observations = memory_observations[:, 0, :, :, :]

        loss = self.network.train_on_batch(memory_observations, memory_targets)
        print('LOSS: {}'.format(loss))

        # copy weights to target network every c iteration and save the network
        if self.update_iteration % self.c_iteration == 0:
            self.target_network.set_weights(self.network.get_weights())
            print('Model Saved!')
            self.network.save_weights(self.path[0])
            self.target_network.save_weights(self.path[1])

        print('UPDATE ITERATION: {}'.format(self.update_iteration))
        if self.update_iteration % 50000 == 0:
            print('===============')
            print('EPOCH: {}'.format(self.update_iteration / 50000))
