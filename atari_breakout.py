import gym
import numpy as np
from agents.dqn import DQNAgent

env = gym.make('Breakout-v4')
agent = DQNAgent(env)

max_episode = 1e7

FRAME_SKIP = 4
batch_size = 32

rewards = []
steps = 1

for episode in range(1, int(max_episode)):
    observation = env.reset()

    terminated = False
    total_reward = 0

    stacked_frame = []
    stacked_frame_action = []
    frame_count = 1
    reward_per_skipped = 0
    stored_action = env.action_space.sample()

    while not terminated:
        # env.render()

        observation = agent.pre_process_observation(observation)
        stacked_frame_action.append(observation)

        # we need 7 observation to create tuples of (x1, x2, x3, x4) for observation and
        # (x2, x3, x4, x5) for new_observation (reference by DeepMind's Nature paper on DQN)
        if frame_count <= FRAME_SKIP * 7:
            action = stored_action
            if frame_count % FRAME_SKIP == 0:
                stacked_observation_action = np.stack(stacked_frame_action[:4], axis=2)
                stacked_observation_action = np.expand_dims(stacked_observation_action, axis=0)
                action = agent.choose_action(stacked_observation_action)
                stored_action = action
                stacked_frame.append(observation)
                stacked_frame_action.clear()
            frame_count += 1
        else:
            stacked_observation = np.stack(stacked_frame[:4], axis=2)
            stacked_new_observation = np.stack(stacked_frame[3:], axis=2)
            stacked_observation = np.expand_dims(stacked_observation, axis=0)
            stacked_new_observation = np.expand_dims(stacked_new_observation, axis=0)
            action = agent.choose_action(stacked_observation)
            stored_action = action
            agent.store_data(stacked_observation, action, reward_per_skipped, stacked_new_observation, terminated)
            stacked_frame.clear()
            frame_count = 1
            reward_per_skipped = 0

        new_observation, reward, terminated, _ = env.step(action)

        total_reward += reward
        reward_per_skipped += reward

        if terminated:
            rewards.append(total_reward)
            if len(rewards) > 100:
                rewards.pop(0)
            print('====================================')
            print('EPSILON: ', agent.epsilon)
            print('FRAMES: ', steps)
            print('AVERAGE REWARD: {}'.format(np.mean(rewards)))
            print('EPISODE: {} FINISHED WITH REWARD: {}'.format(episode, total_reward))
            print('====================================')
            break

        observation = new_observation

        if steps % 4 == 0:
            agent.update_network(steps)
        agent.update_epsilon()
        steps += 1

env.close()
