import open3d as o3d
import numpy as np

# Read the point cloud by loading a np.savez file
point_cloud_data = np.load('pointcloud_real.npz')
point_cloud_points = point_cloud_data['xyz']
point_cloud_colors = point_cloud_data['rgb'] / 255.0

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(point_cloud_points)
point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_colors)
point_cloud_new = point_cloud.voxel_down_sample(voxel_size=0.1)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_new])    