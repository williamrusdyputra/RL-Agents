import threading
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from .agent_utils import build_actor, build_critic

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class A2CAgent:
    def __init__(self, env):
        self.path = ['./weights_a2c/actor/actor', './weights_a2c/critic/critic']
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.shape[0]
        self.n_agent = 4
        self.global_actor = self.build_network('actor')
        self.global_critic = self.build_network('critic')
        self.barrier = threading.Barrier(self.n_agent)
        self.max_time = 1e2
        self.discount = 0.95
        self.state_space_samples = np.array([env.observation_space.sample() for _ in range(10000)])
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.state_space_samples)

    def build_network(self, mode):
        if mode == 'actor':
            model = build_actor(self.action_space, self.state_shape)
            try:
                model.load_weights(self.path[0])
            except ValueError:
                print('Model initialized')
            return model
        elif mode == 'critic':
            model = build_critic(self.state_shape)
            try:
                model.load_weights(self.path[1])
            except ValueError:
                print('Model initialized')
            return model

    def choose_action(self, observation):
        return self.global_actor.predict(observation)

    def create_workers(self):
        workers = []
        for i in range(self.n_agent):
            worker_actor = build_actor(self.action_space, self.state_shape)
            worker_critic = build_critic(self.state_shape)
            workers.append([worker_actor, worker_critic])
        return workers

    def loss_critic(self, model, inputs, targets):
        error = targets - model(inputs)
        return tf.reduce_mean(tf.square(error))

    def loss_actor(self, model, inputs, targets):
        advantage = targets
        return -tf.math.log(model(inputs) + 1e-6) * advantage

    def scale_state(self, state):
        scaled = self.scaler.transform(state)
        return scaled

    def train(self, actor, critic):
        for i in range(20):
            print('{} is running on episode {}'.format(threading.current_thread().name, i+1))
            actor.set_weights(self.global_actor.get_weights())
            critic.set_weights(self.global_critic.get_weights())

            observation = self.env.reset()
            terminated = False

            observations = []
            rewards = []
            actions = []
            time_step = 1
            total_reward = 0

            while not terminated and time_step < self.max_time:
                observation = self.scale_state(observation.reshape(-1, 2))
                observations.append(observation)

                action = actor.predict(observation)

                observation, reward, terminated, _ = self.env.step(action)

                rewards.append(reward)
                actions.append(action)

                total_reward += reward
                time_step += 1

            print('{} finished with reward: {} and {} step'.format(
                threading.current_thread().name, total_reward, time_step))

            if terminated:
                return_estimation = 0
            else:
                observation = self.scale_state(observation.reshape(-1, 2))
                return_estimation = critic.predict(observation)

            actor_gradients = None
            critic_gradients = None
            for j in range(len(observations)):
                return_estimation = self.discount * return_estimation + rewards[j]
                td_error = return_estimation - critic.predict(observations[j])
                with tf.GradientTape(persistent=True) as tape:
                    actor_loss = self.loss_actor(actor, observations[j], td_error)
                    critic_loss = self.loss_critic(critic, observations[j], return_estimation)

                # accumulate gradients w.r.t parameters
                actor_gradient = tape.gradient(actor_loss, actor.trainable_weights)
                if actor_gradients is None:
                    actor_gradients = actor_gradient
                else:
                    actor_gradients += actor_gradient

                critic_gradient = tape.gradient(critic_loss, critic.trainable_weights)
                if critic_gradients is None:
                    critic_gradients = critic_gradient
                else:
                    critic_gradients += critic_gradient

            optimizer = self.global_actor.optimizer
            optimizer.apply_gradients(zip(actor_gradients, self.global_actor.trainable_weights))
            optimizer = self.global_critic.optimizer
            optimizer.apply_gradients(zip(critic_gradients, self.global_critic.trainable_weights))

            print('{} already accumulated gradient. Waiting for other thread..'.format(threading.current_thread().name))
            self.barrier.wait()

    def assign_workers(self):
        workers = self.create_workers()
        threads = []
        for idx, worker in enumerate(workers):
            thread = threading.Thread(target=self.train, args=(worker[0], worker[1]), name='Worker {}'.format(idx+1))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def start_train(self):
        self.assign_workers()
        self.global_actor.save_weights(self.path[0])
        self.global_critic.save_weights(self.path[1])
        print('FINISHED TRAINING')
