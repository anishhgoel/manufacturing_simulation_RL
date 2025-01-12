import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ManufacturingEnv(gym.Env):
    def __init__(self):
        super(ManufacturingEnv, self).__init__()
        #observation space : [temperature, quality]
        self.observation_space = spaces.Box(
            low = np.array([0.0, 0.0]),
            high = np.array([100.0, 100.0]),
            dtype = np.float32
        )

        # actions : 0-> Decrease temp ; 1-> Maintain temp ; 2-> Increase temp

        self.action_space = spaces.Discrete(3)

        self.optimal_temp = 75.0
        self.temp_tolerance = 5.0
        self.state = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_temp = 70 + np.random.normal(0, 2)
        self.state = np.array([initial_temp, 50.0], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        # action -> 0 = temp-2
        # action -> 1 = temp
        # action -> 2 = temp+2

        temp_change = (action-1) * 2
        self.state[0] += temp_change + np.random.normal(0,0.05)  #adding some noise as we cannot have perfect change in temp
        temp_deviation = abs(self.state[0] - self.optimal_temp)

        if temp_deviation <= self.temp_tolerance:
            self.state[1] = min(100, self.state[1] + 0.5)
            reward = 1.0
        else:
            self.state[1] = max(0, self.state[1] - 0.5)
            reward = -1.0

        # will stop if quality if maxed out or completely ruined
        done = bool(self.state[1]<= 0 or self.state[1]>= 100)
        # can add info here if I would want to later on
        info = {}
        return self.state, reward, done, info


        