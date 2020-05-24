import gym
from agents.reinforce import ReinforceAgent

env = gym.make('CartPole-v0')
agent = ReinforceAgent(env)

max_episode = 1000

# solved criteria is average reward the agent gets after t episodes
solved_average_reward = 195
t_episodes = 100

solved_reward_count = 0

for episode in range(1, max_episode):
    observation = env.reset()

    terminated = False
    time_steps = 1
    total_reward = 0

    while not terminated:
        env.render()
        action, prob = agent.choose_action(observation)
        new_observation, reward, terminated, _ = env.step(action)

        total_reward += reward
        agent.store_data(observation, reward, action, prob)

        if terminated:
            print('EPISODE: {} FINISHED AFTER: {} time steps WITH REWARD: {}'.format(episode, time_steps, total_reward))
            break

        observation = new_observation
        time_steps += 1

    agent.update_network()
    agent.reset_data()

    if total_reward >= solved_average_reward:
        solved_reward_count += 1
    else:
        solved_reward_count = 0

    if solved_reward_count >= t_episodes:
        print("AGENT SOLVED THIS PROBLEM! :D")
        break

env.close()
