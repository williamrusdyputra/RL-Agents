import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from .agent_utils import build_network

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
    On first several episodes your agent should already reached the finish line and
    get reward above 80 or 90. If your agent is idle on the middle of the mountain and
    never finish the environment, it means that it got stuck at local optima and you should
    re-run this program.
'''


class A2CAgent:
    def __init__(self, env):
        self.path = ['./weights_a2c/actor/actor', './weights_a2c/critic/critic']
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.shape[0]
        self.global_actor = build_network('actor', self.action_space, self.state_shape, self.path[0])
        self.global_critic = build_network('critic', self.action_space, self.state_shape, self.path[1])
        self.target_critic = build_network('critic', self.action_space, self.state_shape, self.path[1])
        self.max_episode = 40
        self.max_time = 1e3
        self.discount = 0.95
        self.state_space_samples = np.array([env.observation_space.sample() for _ in range(10000)])
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.state_space_samples)

    def choose_action(self, observation):
        observation = self.scale_state(observation.reshape(-1, 2))
        return self.global_actor.predict(observation)

    def loss_actor(self, model, inputs, advantage):
        return -tf.math.log(model(inputs) + 1e-6) * advantage

    def loss_critic(self, model, inputs, targets):
        error = targets - model(inputs)
        return tf.reduce_mean(tf.square(error))

    def scale_state(self, state):
        scaled = self.scaler.transform(state)
        return scaled

    def train(self):
        step = 0
        for i in range(int(self.max_episode)):
            observation = self.env.reset()
            terminated = False

            observations = []
            rewards = []
            actions = []
            time_step = 1
            total_reward = 0

            while not terminated and time_step < self.max_time:
                # self.env.render()

                observation = self.scale_state(observation.reshape(-1, 2))
                observations.append(observation)

                action = self.global_actor.predict(observation)

                observation, reward, terminated, _ = self.env.step(action)

                rewards.append(reward)
                actions.append(action)

                total_reward += reward
                time_step += 1

            if terminated:
                return_estimation = 0
            else:
                observation = self.scale_state(observation.reshape(-1, 2))
                return_estimation = self.target_critic.predict(observation)

            actor_gradients = None
            critic_gradients = None
            for j in range(len(observation)-1, -1, -1):
                return_estimation = self.discount * return_estimation + rewards[j]
                td_error = return_estimation - self.target_critic.predict(observations[j])
                with tf.GradientTape(persistent=True) as tape:
                    actor_loss = self.loss_actor(self.global_actor, observations[j], td_error)
                    critic_loss = self.loss_critic(self.global_critic, observations[j], return_estimation)

                # accumulate gradients w.r.t parameters
                actor_gradient = tape.gradient(actor_loss, self.global_actor.trainable_weights)
                if actor_gradients is None:
                    actor_gradients = actor_gradient
                else:
                    actor_gradients += actor_gradient

                critic_gradient = tape.gradient(critic_loss, self.global_critic.trainable_weights)
                if critic_gradients is None:
                    critic_gradients = critic_gradient
                else:
                    critic_gradients += critic_gradient

            optimizer = self.global_actor.optimizer
            optimizer.apply_gradients(zip(actor_gradients, self.global_actor.trainable_weights))
            optimizer = self.global_critic.optimizer
            optimizer.apply_gradients(zip(critic_gradients, self.global_critic.trainable_weights))

            step += 1

            if step % 10 == 0:
                self.target_critic.set_weights(self.global_critic.get_weights())

            print('Episode {} finished with reward: {} and {} step'.format(
                i+1, total_reward, time_step))

        self.global_actor.save_weights(self.path[0])
        self.global_critic.save_weights(self.path[1])
        print('FINISHED TRAINING')
