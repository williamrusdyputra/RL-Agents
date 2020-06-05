import tensorflow as tf
from collections import deque
from .agent_utils import build_network


class PPOAgent:
    def __init__(self, env):
        self.path = ['./weights_ppo/actor/actor', './weights_ppo/critic/critic']
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        self.actor = build_network('actor', self.action_space, self.state_shape, self.path[0])
        self.critic = build_network('critic', self.action_space, self.state_shape, self.path[1])
        self.old_actor = build_network('actor', self.action_space, self.state_shape, self.path[0])
        self.discount = 0.97
        self.s_lambda = 0.95
        self.epsilon = 0.2
        self.horizon = 256
        self.batch_size = 32
        self.epochs = 10
        self.terminal_step = -1
        self.rewards = deque(maxlen=self.horizon)
        self.observations = deque(maxlen=self.horizon)

    def store_data(self, observation, reward):
        self.observations.append(observation)
        self.rewards.append(reward)

    def choose_action(self, observation):
        return self.actor.predict(observation)

    def loss_objective(self, advantage, reward, observation, new_observation):
        actor_prob = self.actor.predict(observation)
        old_actor_prob = self.old_actor.predict(observation)
        ratio = actor_prob / old_actor_prob
        first_term = ratio * advantage
        second_term = tf.keras.backend.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = tf.math.minimum(first_term, second_term)
        target = reward + self.discount * self.critic.predict(new_observation)
        critic_loss = tf.math.reduce_mean(tf.math.square(target - self.critic.predict(observation)))
        entropy = -tf.reduce_sum(actor_prob * tf.math.log(actor_prob))
        return critic_loss - actor_loss - entropy

    def update_network(self):
        general_advantage_estimations = deque()
        return_estimations = deque()
        time = 0

        for reward in self.rewards[::-1]:
            if time+1 == self.terminal_step:
                return_estimation = reward
                delta = reward - self.critic.predict(self.observations[time])
            else:
                return_estimation = reward + self.discount * self.critic.predict(self.observations[time+1])
                delta = return_estimation - self.critic.predict(self.observations[time])
            gae = delta + self.discount * self.s_lambda * general_advantage_estimations[time+1]
            general_advantage_estimations.append(gae)
            return_estimations.append(return_estimation)
            time += 1

        return_estimations.reverse()
        general_advantage_estimations.reverse()

        # update the actor
        index = 0
        step = self.horizon // self.batch_size
        for i in range(step-1):
            actor_gradients = None
            for j in range(self.batch_size):
                with tf.GradientTape as tape:
                    loss = self.loss_objective(general_advantage_estimations[index], self.rewards[index],
                                               self.observations[index], self.observations[index+1])

                actor_gradient = tape.gradient(loss, self.actor.trainable_weights)
                if actor_gradients is None:
                    actor_gradients = actor_gradient
                else:
                    actor_gradients += actor_gradient

                index += 1

            optimizer = self.actor.optimizer
            optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))

            # update the critic
            self.critic.train_on_batch(self.observations[index-self.batch_size:index],
                                       return_estimations[index-self.batch_size:index])
