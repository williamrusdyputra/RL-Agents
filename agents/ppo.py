import numpy as np
import tensorflow as tf
from collections import deque
from .agent_utils import build_network


class PPOAgent:
    def __init__(self, env):
        self.path = ['./weights_ppo/actor/actor', './weights_ppo/critic/critic']
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.shape[0]
        self.actor = build_network('actor', self.action_space, self.state_shape, self.path[0], 'softmax')
        self.old_actor = build_network('actor', self.action_space, self.state_shape, self.path[0], 'softmax')
        self.critic = build_network('critic', self.action_space, self.state_shape, self.path[1], 'linear')
        self.target_critic = build_network('critic', self.action_space, self.state_shape, self.path[1], 'linear')
        self.discount = 0.99
        self.s_lambda = 0.95
        self.epsilon = 0.1
        self.batch_size = 32
        self.rewards = []
        self.observations = []

    def store_data(self, observation, reward):
        self.observations.append(observation)
        self.rewards.append(reward)

    def choose_action(self, observation):
        return self.actor.predict(observation)[0]

    def loss_objective(self, advantage, reward, observation, new_observation):
        actor_prob = self.actor(observation)
        old_actor_prob = self.old_actor(observation)
        ratio = actor_prob / old_actor_prob
        first_term = ratio * advantage
        second_term = tf.keras.backend.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = tf.math.minimum(first_term, second_term)
        target = reward + self.discount * self.critic(new_observation)
        critic_loss = tf.math.reduce_mean(tf.math.square(target - self.critic(observation)))
        entropy = -tf.reduce_sum(actor_prob * tf.math.log(actor_prob))
        return critic_loss - actor_loss - entropy

    def update_network(self, terminal_step):
        general_advantage_estimations = deque()
        return_estimations = []
        time = 0

        for reward in reversed(self.rewards):
            if time + 1 == terminal_step:
                return_estimation = reward
                delta = reward - self.target_critic.predict(self.observations[time])
            else:
                return_estimation = reward + self.discount * self.target_critic.predict(self.observations[time + 1])
                delta = return_estimation - self.target_critic.predict(self.observations[time])
            if time > 0:
                gae_t1 = general_advantage_estimations[time - 1]
            else:
                gae_t1 = 0
            gae = delta + self.discount * self.s_lambda * gae_t1
            general_advantage_estimations.append(gae)
            return_estimations.append(return_estimation)
            time += 1

        return_estimations.reverse()
        general_advantage_estimations.reverse()

        index = 0
        step = len(self.observations) // self.batch_size
        for i in range(step - 1):
            actor_gradients = None
            for j in range(self.batch_size):
                with tf.GradientTape() as tape:
                    loss = self.loss_objective(general_advantage_estimations[index], self.rewards[index],
                                               self.observations[index], self.observations[index + 1])

                actor_gradient = tape.gradient(loss, self.actor.trainable_weights)
                if actor_gradients is None:
                    actor_gradients = actor_gradient
                else:
                    actor_gradients += actor_gradient

                index += 1

            # update the actor
            optimizer = self.actor.optimizer
            optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))

            # update the critic
            inputs = np.asarray(self.observations[index - self.batch_size:index]).astype('float32')
            outputs = np.asarray(return_estimations[index - self.batch_size:index]).astype('float32')
            inputs = inputs[:, 0, :]
            self.critic.train_on_batch(inputs, outputs)

        # target critic used for handling moving target problem in RL
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor.save_weights(self.path[0])
        self.critic.save_weights(self.path[1])

    def reset_data(self):
        self.observations = []
        self.rewards = []
