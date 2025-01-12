import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ManufacturingEnv(gym.Env):
    def __init__(self, max_steps=300):
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
        self.max_steps = max_steps
        self.state = None
        self.steps = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        initial_temp = 70 + np.random.normal(0, 2.0)
        self.state = np.array([initial_temp, 50.0], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        # action -> 0 = temp-1
        # action -> 1 = temp
        # action -> 2 = temp+1
        self.steps += 1
        temp_change = (action-1) * 1.0
        self.state[0] += temp_change + np.random.normal(0,0.1)  #adding some noise as we cannot have perfect change in temp
        temp_deviation = abs(self.state[0] - self.optimal_temp)

        old_quality = self.state[1]

        if temp_deviation <= self.temp_tolerance:
            self.state[1] = min(100, self.state[1] + 0.5)
        else:
            self.state[1] = max(0, self.state[1] - 0.5)
        new_quality = self.state[1]
        reward = new_quality - old_quality

        # will stop if quality if maxed out or completely ruined
        done = False
        if self.steps >= self.max_steps or self.state[1]<= 0 or self.state[1]>= 100:
            done = True

        # can add info here if I would want to later on
        info = {}
        truncated = (self.steps >= self.max_steps)
        return self.state, reward, done,truncated,  info


        