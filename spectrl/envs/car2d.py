from scipy.stats import truncnorm

import numpy as np
import gym

class VC_Env(gym.Env):
    def __init__(self, time_limit,start_pos=0.0,std=0.5):
        self.start_pos = start_pos
        self.state = None
        self.time_limit = time_limit
        self.time = 0
        self.std = std
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))

    def reset(self):
        self.state = np.array([self.start_pos, 0.0], dtype=np.float32)
        self.time = 0
        return self.state.copy()

    def step(self, action):
        action = action * np.array([0.5, np.pi]) + np.array([0.5, 0.])
        velocity = action[0] * np.array([np.cos(action[1]), np.sin(action[1])])
        next_state = self.state + velocity + truncnorm.rvs(-1, 1, 0, self.std, 2)
        self.state = next_state
        self.time = self.time + 1
        return next_state, 0, self.time > self.time_limit, None

    def render(self):
        pass

    def get_sim_state(self):
        return self.state

    def set_sim_state(self, state):
        self.state = state
        return self.state

    def close(self):
        pass