import os
import numpy as np
from sim import LidarSim

LOG_DIR = "logs/"

if __name__ == '__main__':
    
    # log dir
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # generate data
    num_envs = 10 # total number of environments
    num_steps = 50 # number of actions per env
    idx = 0
    
    for env_it in range(num_envs):
        # init simulator
        grid_width = 100 # 100 meters
        grid_height = 20 # 20 meters
        voxel_resolution = 0.2        
        sim = LidarSim(grid_width, grid_height, voxel_resolution=0.2, h_res=90, v_res=45, h_fov_deg=360, v_fov_deg=45)
        # create a scene with random cylinders and ground plane
        scene = sim.create_scene(num_cylinders=10)
        # random initial pose
        old_pose = np.array([np.random.uniform(10, 90), np.random.uniform(1, 5), np.random.uniform(10, 90), 0, 0, 0])
        for i in range(num_steps):
            # sample new pose within some distance of old pose, x and z within 5 and 95, y within 1 and 15
            x_old, y_old, z_old = old_pose[0], old_pose[1], old_pose[2]
            # height
            y_low = max(1, y_old - 5)
            y_high = min(15, y_old + 5)
            # width
            x_low = max(5, x_old - 10)
            x_high = min(95, x_old + 10)
            z_low = max(5, z_old - 10)
            z_high = min(95, z_old + 10)
            x_new, y_new, z_new = np.random.uniform(low=[x_low, y_low, z_low], 
                                                    high=[x_high, y_high, z_high])
            new_pose = np.array([x_new, y_new, z_new])
            # print(new_pose)
                            
            voxel_grid_old = sim.get_o3d_voxel_grid()
            sim.simulate_step(new_pose)
            voxel_grid = sim.get_o3d_voxel_grid()

            # sim.visualize(False, True, False) # point cloud, voxel grid, scene

            if voxel_grid_old is not None:
                # skip the first action
                # save pose, initial voxel_grid, and updated voxel_grid to file
                old_pose.tofile(LOG_DIR+'pose_old_{}.bin'.format(idx))
                new_pose.tofile(LOG_DIR+'pose_new_{}.bin'.format(idx))
                point_cloud_old = np.asarray([voxel_grid_old.origin + voxel_resolution/2.0 + pt.grid_index*voxel_grid_old.voxel_size for pt in voxel_grid_old.get_voxels()])
                point_cloud_old.tofile(LOG_DIR+'grid_old_{}.bin'.format(idx))
                point_cloud_new = np.asarray([voxel_grid.origin + voxel_resolution/2.0 + pt.grid_index*voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])
                point_cloud_new.tofile(LOG_DIR+'grid_new_{}.bin'.format(idx))
                idx += 1
    