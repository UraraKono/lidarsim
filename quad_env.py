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
        self.observation_space = Box(low=np.array([0]*3), high=np.array([100/0.02]*3)) # will need to edit later - value between 0 and 100
        self.state_x = random.randint(0, 100) #initial x width index
        self.state_y = random.randint(0, 100) #intial y height index
        self.state_z = 2 #initial z meters above ground
        self.length = 20 #max # of timesteps
        self.sim = None
        self.sum_reward = 0

        self.occupied_indices = None
        self.found_voxels = None

        self.grid_width = 100 # 100 meters
        self.grid_height = 20 # 20 meters
        #self.scene = sim.create_scene(num_cylinders=num_cyl)

        random.seed(1)

        #ground truth occupied voxel grid
        global_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(sim.get_scene(), voxel_size=0.2) #converts scene from o3d mesh to voxel grid
        voxels = global_voxel_grid.get_voxels()  # list of voxels in the grid
        occupied_indices = np.stack(list(vx.grid_index for vx in voxels)) # numpy array of occupied voxels

    def get_pose(self):
        # return the pose of quad for open3d xzy
        return np.array([self.state_x, self.state_z, self.state_y])

    def step(self, action, eps = 0):
        reward = 0
        # print('length',self.length)
        #print('t_remain',self.t_remain)
        print(self.t_remain)
        if self.t_remain == self.length:
        #   print('initialize self.voxel_grid')
            self.sim.simulate_step(self.get_pose())
        voxel_grid_old = self.sim.get_o3d_voxel_grid()
        voxel_grid_old_list = voxel_grid_old.get_voxels()
        # grid_old = self.sim.get_numpy_voxel_grid()
        # print('voxel_grid_old_ printing',voxel_grid_old)
        # print('grid_old',grid_old)

        #epsilon-greedy learning
        if eps > random.random():
            action = self.action_space.sample()
        else:
            action = action

        #update state
        # move by 1 meter, 1/self.sim.voxel_resolution = 5 voxels
        #0 - stay still
        if action < 4 and action > 0:
            self.state_y += 1
        if action < 6 and action > 2:
            self.state_x += 1
        if action < 8 and action > 4:
            self.state_y -= 1
        if action > 6 or action == 1:
            self.state_x -= 1

        # clip the state to be within the grid
        self.state_x = np.clip(self.state_x, 0, self.sim.grid_width)
        self.state_y = np.clip(self.state_y, 0, self.sim.grid_width)
        self.state_z = np.clip(self.state_z, 0, self.sim.grid_height)

        #simulate step in sim to get observation
        self.sim.simulate_step(self.get_pose())
        # Get the voxel grid given the new observation
        # grid = self.sim.get_numpy_voxel_grid() # numpy 3d array of 0 or 1 for each voxel
        voxel_grid = self.sim.get_o3d_voxel_grid()
        voxel_grid_list = voxel_grid.get_voxels()

        prev_points = self.found_voxels.shape[0]

        #print(self.found_voxels.shape)
        #print(np.stack(list(vx.grid_index for vx in voxel_grid_list)).shape)

        self.found_voxels = np.vstack((self.found_voxels, np.stack(list(vx.grid_index for vx in voxel_grid_list))))
        self.found_voxels = np.unique(self.found_voxels, axis=0)
        # print('voxel_grid',voxel_grid) # None!!!! Why is it none? 
        # print('grid',grid) 
        
        curr_points = self.found_voxels.shape[0]
        print(curr_points)
        print(prev_points)

        #update time steps remaining
        self.t_remain -= 1

        #TODO - update reward

        # number of points that are in both the old and new voxel grid
        # points_old_now_count = 0
        # for point in voxel_grid_old_list:
        #     if point in voxel_grid_list:
        #         points_old_now_count += 1

        # info gain = # of newly observed voxels
        #info_gain = len(voxel_grid_list) - points_old_now_count
        info_gain = curr_points - prev_points
        self.n_observed_voxels += info_gain

        # TODO
        # If the current state is in occupied voxel grid, action cost = 100
        # If we cover the 90% of the entire map, action cost = 0.01
        # We have reward 
        action_cost = 1
        reward = info_gain / action_cost
        self.sum_reward += reward

        #update done state
        if self.t_remain <= 0:
            done = True
        else:
            done = False

        #TODO - condition when voxel grid is 30% mapped
        # We have ground truth voxel grid, so we can compare the two
        if self.n_observed_voxels / self.total_voxels >= 0.9:
            done = True
            print('Covered 90% of the map!')
            info = {'Time' : self.t_remain}

        info = {'Time' : self.t_remain, 'Done': done, 'Info Gain': info_gain, 'Reward': reward, \
                'Action': action, 'State x': self.state_x, 'State y': self.state_y, 'Sum Reward': self.sum_reward}

        print('reward: ', reward)

        # Return step information
        return self.get_pose().astype(np.float32), reward, done, info

    def reset(self):
        self.state_x = random.randint(0, 100) #initial x width index
        self.state_y = random.randint(0, 100) #intial y height index
        self.state_z = 2  # initial z meters above ground
        self.length = 10
        self.t_remain = self.length
        self.sum_reward = 0

        #creats a new scene every time
        self.sim = LidarSim(100, 20, voxel_resolution=0.2, h_res=90, v_res=45, h_fov_deg=360, v_fov_deg=45)
        self.scene = self.sim.create_scene(num_cylinders=10)

        #ground truth occupied voxel grid
        global_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self.sim.get_scene(), voxel_size=0.2) #converts scene from o3d mesh to voxel grid
        voxels = global_voxel_grid.get_voxels()  # list of voxels in the grid #properly resetting

        #ground truth
        self.occupied_indices = np.stack(list(vx.grid_index for vx in voxels)) # numpy array of occupied voxels
        print(type(self.occupied_indices))
        #current map
        self.found_voxels = np.zeros((1, 3)) #[0, 0, 0] will always be 'found'
        self.total_voxels = len(voxels) #total number of occupied voxels in the ground truth
        self.n_observed_voxels = 0 #number of observed voxels

        # TODO - ground truth occupied voxel grid on cylinders
        self.scene_cylinders = self.sim.create_scene(num_cylinders=10, create_ground=False)


        # print(global_voxel_grid) # VoxelGrid with 34038 voxels.
        # print('total voxels',self.total_voxels) # total voxels 34038
        # print('occupied_indices',occupied_indices)
        print("RESET")

        # return the observation
        return self.get_pose().astype(np.float32)

    #TODO - render model
    def render(self):
        print(self.found_voxels)
        print(self.occupied_indices)
        #print ground truth voxels
        #print found voxels