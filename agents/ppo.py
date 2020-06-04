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
        self.rewards = deque(maxlen=128)
        self.observations = deque(maxlen=256)

    def choose_action(self, observation):
        return self.actor.predict(observation)

    def loss_objective(self, inputs, advantage, reward, observation, new_observation):
        actor_prob = self.actor.predict(inputs)
        old_actor_prob = self.old_actor.predict(inputs)
        ratio = actor_prob / old_actor_prob
        first_term = ratio * advantage
        second_term = tf.keras.backend.clip(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
        actor_loss = tf.math.minimum(first_term, second_term)
        target = reward + self.discount * self.critic.predict(new_observation)
        critic_loss = tf.math.reduce_mean(tf.math.square(target - self.critic.predict(observation)))
        entropy = -tf.reduce_sum(actor_prob * tf.math.log(actor_prob))
        return critic_loss - actor_loss - entropy

    def update_network(self, terminal_step):
        general_advantage_estimations = []
        time = 0

        for reward in self.rewards[::-1]:
            if time+1 == terminal_step:
                delta = reward - self.critic.predict(self.observations[time])
            else:
                delta = (reward + self.discount * self.critic.predict(self.observations[time+1]) -
                         self.critic.predict(self.observations[time]))
            gae = delta + self.discount * self.s_lambda * general_advantage_estimations[time+1]
            general_advantage_estimations.append(gae)
            time += 1

        general_advantage_estimations.reverse()
