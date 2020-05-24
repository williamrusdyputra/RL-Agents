import gym
from agents.dqn import DQNAgent

env = gym.make('Breakout-v0')
agent = DQNAgent(env)

max_episode = 1000

for episode in range(1, max_episode):
    observation = env.reset()

    terminated = False
    time_steps = 1
    total_reward = 0

    while not terminated:
        env.render()
        action = agent.choose_action(observation)
        new_observation, reward, terminated, _ = env.step(action)

        total_reward += reward
        agent.store_data(observation, action, reward, new_observation, terminated)

        if terminated:
            print('EPISODE: {} FINISHED AFTER: {} time steps WITH REWARD: {}'.format(episode, time_steps, total_reward))
            break

        observation = new_observation
        time_steps += 1

env.close()
