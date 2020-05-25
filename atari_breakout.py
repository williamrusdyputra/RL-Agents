import gym
import numpy as np
from agents.dqn import DQNAgent

env = gym.make('Breakout-v0')
agent = DQNAgent(env)

max_episode = 1000

for episode in range(1, max_episode):
    observation = env.reset()

    terminated = False
    time_steps = 1
    total_reward = 0

    FRAME_SKIP = 4
    stacked_frame = []
    frame_count = 0
    reward_per_skipped = 0

    while not terminated:
        env.render()

        observation = agent.pre_process_observation(observation)

        # we need 7 observation to create tuples of (x1, x2, x3, x4) for observation and
        # (x2, x3, x4, x5) for new_observation (reference by DeepMind's Nature paper on DQN)
        if frame_count < FRAME_SKIP * 7:
            action = env.action_space.sample()
            # DeepMind's frame skipping technique with k = 4
            if frame_count % FRAME_SKIP == 0:
                stacked_frame.append(observation)
            frame_count += 1
        else:
            stacked_observation = np.stack(stacked_frame[:4], axis=2)
            stacked_new_observation = np.stack(stacked_frame[3:], axis=2)
            stacked_observation = np.expand_dims(stacked_observation, axis=0)
            stacked_new_observation = np.expand_dims(stacked_new_observation, axis=0)
            action = agent.choose_action(stacked_observation)
            agent.store_data(stacked_observation, action, reward_per_skipped, stacked_new_observation, terminated)
            stacked_frame.clear()
            frame_count = 0
            reward_per_skipped = 0

        new_observation, reward, terminated, _ = env.step(action)

        total_reward += reward
        reward_per_skipped += reward

        if terminated:
            print('EPISODE: {} FINISHED WITH REWARD: {}'.format(episode, total_reward))
            break

        observation = new_observation
        time_steps += 1

env.close()
