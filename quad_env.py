import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
from sim import LidarSim
import open3d as o3d

#tutorial: https://colab.research.google.com/drive/1oBe07b28h9GCBy_bKtLJisC98mayDcwn?usp=sharing#scrollTo=ns6duy9JDP3S

class QuadEnv(Env):
    def __init__(self, sim, num_cyl = 10):
        self.action_space = Discrete(9)
        self.observation_space = Box(low=np.array([0]*3), high=np.array([100]*3)) # will need to edit later - value between 0 and 100
        self.state_x = random.randint(0, 500) #initial x (width/voxel grid)
        self.state_y = random.randint(0, 500) #intial y (length/voxel grid)
        self.state_z = 10 #initial z meters above ground
        self.length = 100 #max # of timesteps
        self.sim = sim
        self.scene = sim.create_scene(num_cylinders=num_cyl)

        #TODO - ground truth occupied voxel grid
        global_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(sim.get_scene(), voxel_size=0.2) #converts scene from o3d mesh to voxel grid
        voxels = global_voxel_grid.get_voxels()  # list of voxels in the grid
        occupied_indices = np.stack(list(vx.grid_index for vx in voxels)) # numpy array of occupied voxels

    def step(self, action, eps = 0):
        voxel_grid_old = self.sim.get_o3d_voxel_grid()

        #epsilon-greedy learning
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

        #simulate step in sim to get observation
        self.sim.simulate_step(np.array([self.state_x, self.state_y, self.state_z]))
        voxel_grid = self.sim.get_o3d_voxel_grid()
        # based on the action and the observation, we get the reward

        #update time steps remaining
        self.length -= 1

        #TODO - update reward
        # If the current state is in occupied voxel grid, reward = -100
        # If we cover the 90% of the entire map, reward = 1000
        # We have reward 
        reward = 0

        #update done state
        if self.length <= 0:
            done = True
        else:
            done = False

        #TODO - condition when voxel grid is 90% mapped
        # We have ground truth voxel grid, so we can compare the two

        info = {}

        # Return step information
        return self.state, reward, done, info

    def reset(self):
        self.state_x = random.randint(0, 500) #initial x (width/voxel grid)
        self.state_y = random.randint(0, 500) #intial y (length/voxel grid)
        self.state_z = 10  # initial z meters above ground
        self.length = 60
        self.scene = self.sim.create_scene(num_cylinders=10)
        # TODO - ground truth voxel grid
        return self.state



