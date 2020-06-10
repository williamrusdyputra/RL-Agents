import torch
import torch.nn as nn
import tensorflow as tf
from torch.distributions import MultivariateNormal


def build_network(mode, action_space, observation_shape, path, output_activation):
    if mode == 'actor':
        model = build_actor(action_space, observation_shape, output_activation)
        try:
            model.load_weights(path)
            print('Model loaded')
        except ValueError:
            print('Model initialized')
        return model
    elif mode == 'critic':
        model = build_critic(observation_shape)
        try:
            model.load_weights(path)
            print('Model loaded')
        except ValueError:
            print('Model initialized')
        return model


def build_actor(action_space, observation_shape, activation):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='tanh', input_shape=observation_shape),
        tf.keras.layers.Dense(units=64, activation='tanh'),
        tf.keras.layers.Dense(units=action_space, activation=activation)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))

    return model


def build_critic(observation_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='tanh', input_shape=observation_shape),
        tf.keras.layers.Dense(units=512, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.log_probs = []
        self.rewards = []
        self.terminals = []

    def reset(self):
        self.actions = []
        self.observations = []
        self.log_probs = []
        self.rewards = []
        self.terminals = []


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, action_std):
        super(ActorCritic, self).__init__()
        self.agent_action = torch.full((action_space,), action_std * action_std).to(device)
        self.actor = nn.Sequential(
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_space, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def choose_action(self, observation, memory):
        action_mean = self.actor(observation)
        cov_mat = torch.diag(self.agent_action).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        memory.observations.append(observation)
        memory.actions.append(action)
        memory.log_probs.append(action_log_prob)

        return action.detach()

    def evaluate(self, observation, action):
        action_mean = self.actor(observation)

        agent_action = self.agent_action.expand_as(action_mean)
        cov_mat = torch.diag_embed(agent_action).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        observation_value = self.critic(observation)

        return action_log_probs, torch.squeeze(observation_value), dist_entropy
