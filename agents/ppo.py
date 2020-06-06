import torch
import torch.nn as nn
import numpy as np
from .agent_utils import ActorCritic

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PPOAgent:
    def __init__(self, env):
        self.learning_rate = 3e-4
        self.discount = 0.99
        self.clip = 0.2
        self.epochs = 100
        self.std = 0.4
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.MseLoss = nn.MSELoss()

        self.policy = ActorCritic(self.state_space, self.action_space, self.std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.old_policy = ActorCritic(self.state_space, self.action_space, self.std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

    def choose_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.old_policy.choose_action(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.discount * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = np.float32(rewards)
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_log_probs = torch.squeeze(torch.stack(memory.log_probs), 1).to(device).detach()

        for _ in range(self.epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratio = torch.exp(log_probs - old_log_probs.detach())
            advantages = rewards - state_values.detach()
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            loss = -torch.min(surrogate1, surrogate2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
