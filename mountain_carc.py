import gym
from agents.a2c import A2CAgent

env = gym.make("MountainCarContinuous-v0")
agent = A2CAgent(env)

agent.train()

# testing the agent
observation = env.reset()
terminated = False
total_reward = 0
step = 1

while not terminated:
    env.render()
    action = agent.choose_action(observation)
    observation, reward, terminated, _ = env.step(action)
    total_reward += reward
    step += 1

    if terminated:
        print("TESTING FINISHED WITH REWARD: {} and {} step".format(total_reward, step))
        break
