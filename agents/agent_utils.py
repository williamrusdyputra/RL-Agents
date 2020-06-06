import torch
import torch.nn as nn
import tensorflow as tf
from torch.distributions import MultivariateNormal


def build_network(mode, action_space, state_shape, path, output_activation):
    if mode == 'actor':
        model = build_actor(action_space, state_shape, output_activation)
        try:
            model.load_weights(path)
            print('Model loaded')
        except ValueError:
            print('Model initialized')
        return model
    elif mode == 'critic':
        model = build_critic(state_shape)
        try:
            model.load_weights(path)
            print('Model loaded')
        except ValueError:
            print('Model initialized')
        return model


def build_actor(action_space, state_shape, activation):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='tanh', input_shape=state_shape),
        tf.keras.layers.Dense(units=64, activation='tanh'),
        tf.keras.layers.Dense(units=action_space, activation=activation)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))

    return model


def build_critic(state_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='tanh', input_shape=state_shape),
        tf.keras.layers.Dense(units=512, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space, action_std):
        super(ActorCritic, self).__init__()
        self.agent_action = torch.full((action_space,), action_std * action_std).to(device)
        self.actor = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def choose_action(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.agent_action).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.log_probs.append(action_log_prob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        agent_action = self.agent_action.expand_as(action_mean)
        cov_mat = torch.diag_embed(agent_action).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_log_probs, torch.squeeze(state_value), dist_entropy
