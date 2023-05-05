import numpy as np
import open3d as o3d
from sim import LidarSim

if __name__ == '__main__':
    # init simulator
    grid_width = 100 # 100 meters
    grid_height = 20 # 20 meters
    voxel_resolution = 0.2 # 0.2 meters
    sim = LidarSim(grid_width, grid_height, voxel_resolution, h_res=90, v_res=45, h_fov_deg=360, v_fov_deg=45)
    
    # create a scene with random cylinders and ground plane
    sim.create_scene(num_cylinders=10, create_ground=True)

    for i in range(10):
        # define the pose in the scene (x, z, y, yaw, pitch, roll)
        pose = np.array([np.random.uniform(0, 100), np.random.uniform(1, 5), np.random.uniform(0, 100), 0, 0, 0])
        
        # simulate a lidar scan and update the voxel grid
        sim.simulate_step(pose)
        
        # visualize voxel grid overlayed on scene
        sim.visualize(False, True, True) # point cloud, voxel grid, scene
        
        # get numpy voxel grid
        grid = sim.get_numpy_voxel_grid()
        print('grid in example.py',grid)
        
        # visualize numpy voxel grid overlayed on o3d voxel grid
        pcd = o3d.geometry.PointCloud()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    if grid[i,j,k] == 1:
                        pcd.points.append(np.array((i * 0.2 + 0.1, k * 0.2 + 0.1, j * 0.2 + 0.1))) # o3d is xzy
        # visualize point cloud overlayed on voxel grid
        o3d.visualization.draw_geometries([pcd, sim.voxel_grid])
