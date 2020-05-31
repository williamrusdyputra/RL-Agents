from .agent_utils import build_actor, build_critic


class A2CAgent:
    def __init__(self, env):
        self.path = ['./weights_a2c/actor/actor', './weights_a2c/critic/critic']
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        self.n_agent = 5
        self.global_actor = build_actor()
        self.global_critic = build_critic()
