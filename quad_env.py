import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
from stable_baselines3.common.env_checker import check_env

#tutorial: https://colab.research.google.com/drive/1oBe07b28h9GCBy_bKtLJisC98mayDcwn?usp=sharing#scrollTo=ns6duy9JDP3S

class QuadEnv(Env):
    def __init__(self):
        self.action_space = Discrete(9)
        self.observation_space = Box(low=np.array([0]), high=np.array([100])) # will need to edit later - value between 0 and 100
        self.state_x = random.randint(0, 500) #initial x (width/voxel grid)
        self.state_y = random.randint(0, 500) #intial y (length/voxel grid)
        self.length = 100 #episode length

    def step(self, action, eps = 0):
        #epsilon learning
        if eps > random.random():
            action = self.action_space.sample()
        else:
            action = action

        #update state
        #0 - stay still
        if action < 4 and action > 0:
            self.state_y += 1
        if action < 6 and action > 2:
            self.state_x += 1
        if action < 8 and action > 4:
            self.state_y -= 1
        if action > 6 or action == 1:
            self.state_x -= 1

        #update time steps remaining
        self.length -= 1

        #update reward
        reward = 0
        # if self.state >= 37 and self.state <= 39:
        #     reward = 1
        # else:
        #     reward = -1

        #update done state
        if self.length <= 0:
            done = True
        else:
            done = False

        #condition when voxel grid is 90% mapped

        info = {}

        # Return step information
        return self.state, reward, done, info

    def reset(self):
        self.state_x = random.randint(0, 500) #initial x (width/voxel grid)
        self.state_y = random.randint(0, 500) #intial y (length/voxel grid)
        self.length = 60
        return self.state



