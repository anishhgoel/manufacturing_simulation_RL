import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class ManufacturingEnv(gym.Env):
    def __init__(self, max_steps=300, temp_optimal=75.0, pressure_optimal=50.0, flow_optimal=20.0, anomaly_prob=0.01):
        super(ManufacturingEnv, self).__init__()
        #observation space : [temperature,pressure, flow, quality]
        self.observation_space = spaces.Box(
            low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([100.0, 100.0, 50.0, 100.0], dtype = np.float32),
            dtype = np.float32
        )

        # actions : 0-> Decrease temp ; 1-> Maintain temp ; 2-> Increase temp

        self.action_space = spaces.Discrete(3)

        self.temp_optimal = temp_optimal
        self.pressure_optimal = pressure_optimal
        self.fow_optimal = flow_optimal
        self.max_steps = max_steps
        self.steps = 0

        self.anomaly_prob = anomaly_prob # to account for random anamaly events
        self.state = None
        self.temp_tolerance = 5
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        initial_temp = 70 + np.random.normal(0, 2.0)
        initial_pressure = 50.0 + np.random.normal(0,2.0)
        initial_flow = 20.0 + np.random.normal(0,1.0)
        initial_quality = 50.0
        self.state = np.array([initial_temp, initial_pressure, initial_flow, initial_quality], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        # action -> 0 = temp-1
        # action -> 1 = temp
        # action -> 2 = temp+1
        self.steps += 1
        temp, pressure, flow, quality = self.state

        # temperature control and adding noises
        temp_change = (action-1) * 1.0
        self.state[0] += temp_change + np.random.normal(0,0.1)  #adding some noise as we cannot have perfect change in temp
        pressure += np.random.normal(0,0.05)
        flow = np.random.normal(0,0.02)

        #cliiping into vlaid ranges
        temp = np.clip(temp, 0, 100)
        pressure = np.clip(pressure, 0, 100)
        flow = np.clip(flow, 0, 50)

        #accounting for random anamoly or breakdoen (1% chance that temperature spikes up or down in range +- 10)
        if np.random.rand() < self.anomaly_prob:
            anamoly_change = np.random.choice([-10.0, 10.0])
            temp += anamoly_change
            temp = np.clip(temp, 0, 100)

        #using gaussian based approach

        sigma_temp = 5.0
        sigma_pressure = 5.0
        sigma_flow = 3.0

        temp_term = math.exp(-((temp - self.temp_optimal) ** 2) / (2 * sigma_temp ** 2))
        press_term = math.exp(-((pressure - self.pressure_optimal) ** 2) / (2 * sigma_pressure ** 2))
        flow_term = math.exp(-((flow - self.flow_optimal) ** 2) / (2 * sigma_flow ** 2))
        
        #combining for simplicity. Can also do weighted depending on requirements specific to customer/domain.**** Will do later
        new_quality = 100.0 * (temp_term * press_term * flow_term)
        new_quality = np.clip(new_quality, 0, 100)

        reward_from_quality = new_quality - quality
        #accoiunting for energy cost per temperature change
        energy_cost = -0.1 * abs(temp_change)

        reward = reward_from_quality + energy_cost
        #new state
        self.state = np.array([temp, pressure, flow, new_quality], dtype=np.float32)

        
        done = False
        if self.steps >= self.max_steps or new_quality <= 0 or new_quality >= 100:
            done = True
        truncated = (self.steps >= self.max_steps)
        
        info = {}

        return self.state, reward, done, truncated, info


        