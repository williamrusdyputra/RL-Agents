import gym
from agents.ppo import PPOAgent

env = gym.make("Hopper-v2")
agent = PPOAgent(env)

max_episode = int(1e6)

for episode in range(1, max_episode):
    observation = env.reset()

    terminated = False
    time_steps = 1
    total_reward = 0

    while not terminated:
        # env.render()
        action = agent.choose_action(observation)
        new_observation, reward, terminated, _ = env.step(action)

        total_reward += reward
        agent.store_data(observation, reward)

        if terminated:
            print('EPISODE: {} FINISHED AFTER: {} time steps WITH REWARD: {}'.format(episode, time_steps, total_reward))
            break

        observation = new_observation
        time_steps += 1

    agent.update_network()

env.close()
