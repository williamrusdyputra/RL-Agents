import gym
from agents.a2c import A2CAgent

env = gym.make("MountainCarContinuous-v0")
agent = A2CAgent(env)

agent.start_train()

# testing the agent
observation = env.reset()
terminated = False
total_reward = 0

while not terminated:
    env.render()
    observation = agent.scale_state(observation.reshape(-1, 2))
    action = agent.choose_action(observation)

    observation, reward, terminated, _ = env.step(action)

    total_reward += reward

    if terminated:
        print("TESTING FINISHED WITH REWARD: {}".format(total_reward))
        break
