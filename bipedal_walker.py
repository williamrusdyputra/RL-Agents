import gym
import torch
from agents.ppo import PPOAgent
from agents.agent_utils import Memory

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

max_episode = 1e3
max_steps = 1e3

UPDATE_STEP = 2e3

env = gym.make("BipedalWalker-v3")

memory = Memory()
agent = PPOAgent(env)

total_reward = 0
time_step = 0

for i in range(1, int(max_episode)):
    observation = env.reset()
    total_reward = 0
    for _ in range(int(max_steps)):
        env.render()

        action = agent.choose_action(observation, memory)
        observation, reward, terminated, _ = env.step(action)

        time_step += 1

        memory.rewards.append(reward)
        memory.is_terminals.append(terminated)

        if time_step % UPDATE_STEP == 0:
            agent.update(memory)
            memory.clear_memory()
            time_step = 0
            print('AGENT UPDATED')
        total_reward += reward
        if terminated:
            break

    print('EPISODE {} COMPLETED WITH REWARD: {}'.format(i, total_reward))

    if i % 200 == 0:
        torch.save(agent.policy.state_dict(), './weights_ppo.pth')
