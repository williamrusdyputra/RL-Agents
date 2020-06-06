# RL-Agents

Various of Reinforcement Learning agents. Tested on OpenAI Gym environments.
Repository was made because without implementing RL agents by myself, it is hard to grasp how the algorithms really work.

Current agents that are available:
  - **REINFORCE (Monte Carlo Policy Gradient)** [*done, cart-pole with average reward 500*] using Tensorflow
  - **A2C (Advantage Actor-Critic)** [*done, mountain car continuous with average reward 96*] using Tensorflow
  - **DQN (Deep Q-Networks)** [*on progress*]
  - **PPO (Proximal Policy Optimization)** [*on progress*] using PyTorch
  
I used both Tensorflow and PyTorch because it is hard to make custom changes to Tensorflow Keras API and thus I use PyTorch.
I am shifting from Tensorflow to PyTorch after implementing PPO because PyTorch is better IMO to make custom changes.
