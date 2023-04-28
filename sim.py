import open3d as o3d
import numpy as np
import trimesh
import time

class LidarSim:
    def __init__(self, grid_width=100, grid_height=20, voxel_resolution=0.2, h_res=90, v_res=45, h_fov_deg=360, v_fov_deg=45, verbose=False):
        self.grid_width = float(grid_width)
        self.grid_height = float(grid_height)
        self.voxel_resolution = float(voxel_resolution)
        self.h_res = h_res
        self.v_res = v_res
        self.h_fov_deg = h_fov_deg
        self.v_fov_deg = v_fov_deg
        self.verbose = verbose
        
        self.voxel_grid = None
        
    def create_random_cylinder(self):
        radius = np.random.uniform(0.5, 3)
        height = np.random.uniform(3, 15)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, height)

        # rotate the cylinder to be upright
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2.0, 0, 0))
        cylinder.rotate(rotation)
        
        # random location
        tx, tz = np.random.uniform(0, 100, 2)
        ty = height / 2
        translation = np.array([tx, ty, tz])
        cylinder.translate(translation)

        return cylinder

    def create_ground_plane(self, size=100):
        ground_plane = o3d.geometry.TriangleMesh.create_box(width=size, height=0.01, depth=size)
        return ground_plane

    def create_scene(self, num_cylinders=10, create_ground=True):
        if create_ground:
            self.scene = self.create_ground_plane(self.grid_width)

        for _ in range(num_cylinders):
            cylinder = self.create_random_cylinder()
            self.scene += cylinder

    def simulate_lidar(self, pose):
        if self.scene is None:
            raise Exception("Scene not initialized")
            
        origin = pose[:3]
        mesh = trimesh.Trimesh(vertices=np.asarray(self.scene.vertices), faces=np.asarray(self.scene.triangles))
        rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        h_fov_rad = np.radians(self.h_fov_deg)
        v_fov_rad = np.radians(self.v_fov_deg)
        h_angles = np.linspace(-h_fov_rad / 2, h_fov_rad / 2, self.h_res)
        v_angles = np.linspace(-v_fov_rad / 2, v_fov_rad / 2, self.v_res)

        point_cloud = []

        for h_angle in h_angles:
            for v_angle in v_angles:
                direction = np.array([
                    np.cos(v_angle) * np.cos(h_angle),
                    np.sin(v_angle),
                    np.cos(v_angle) * np.sin(h_angle)
                ])

                ray_origins = np.array([origin])
                ray_directions = np.array([direction])
                locations, index_ray, index_tri = rmi.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)

                if len(locations) > 0 and np.linalg.norm(locations[0] - origin) < 15.0:
                    point_cloud.append(locations[0])
        point_cloud = np.array(point_cloud)
        
        return np.array(point_cloud)

    def visualize_point_cloud(self, point_cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        o3d.visualization.draw_geometries([pcd])

    def simulate_step(self, pose):
        # Generate a point cloud simulating a lidar sensor and update the voxel grid
        start_time = time.time()
        # get new point cloud
        point_cloud = self.simulate_lidar(pose)
        # add both grid and new point cloud
        pcd = o3d.geometry.PointCloud()
        if point_cloud.shape[0] > 0:
            if self.voxel_grid is not None:
                point_cloud_grid = np.asarray([self.voxel_grid.origin + self.voxel_resolution/2.0 + pt.grid_index*self.voxel_grid.voxel_size for pt in self.voxel_grid.get_voxels()])
                pcd.points = o3d.utility.Vector3dVector(np.concatenate((point_cloud_grid, point_cloud), axis=0))
            else:
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
            self.point_cloud = pcd
            self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.voxel_resolution)
        
        end_time = time.time()
        if self.verbose:
            print("simulation time: {:.2f}s".format(end_time - start_time))
    
    def get_scene(self):
        return self.scene

    def visualize(self, viz_cloud=False, viz_grid=True, viz_scene=True):
        geoms = []
        if viz_cloud:
            geoms.append(self.point_cloud)
        if viz_grid:
            geoms.append(self.voxel_grid)
        if viz_scene:
            geoms.append(self.scene)
        o3d.visualization.draw_geometries(geoms)
        
    def visualize_point_cloud(self):
        o3d.visualization.draw_geometries([self.point_cloud])
        
    def get_o3d_voxel_grid(self):
        return self.voxel_grid
    
    def get_numpy_voxel_grid(self):
        voxel_grid_np = np.zeros((int(self.grid_width/self.voxel_resolution), int(self.grid_width/self.voxel_resolution), int(self.grid_height/self.voxel_resolution)))
        # get voxel centers as points
        point_cloud_grid = np.asarray([self.voxel_grid.origin + self.voxel_resolution/2.0 + pt.grid_index*self.voxel_grid.voxel_size for pt in self.voxel_grid.get_voxels()])
        # set occupancy in voxel grid
        for pt in point_cloud_grid:
            i = np.min([int(pt[0]/self.voxel_resolution), voxel_grid_np.shape[0]-1])
            j = np.min([int(pt[1]/self.voxel_resolution), voxel_grid_np.shape[2]-1])
            k = np.min([int(pt[2]/self.voxel_resolution), voxel_grid_np.shape[1]-1])            
            voxel_grid_np[i,k,j] = 1
            
        return voxel_grid_np

        