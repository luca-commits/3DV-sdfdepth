import numpy as np
import os 
import json
import cv2 as cv

jsonfile = "/cluster/home/nihars/mdeNeRF/newgroup/nihars_tests/kitti/datasets_cvpr/2011_09_26_drive_0001_sync_0/transforms.json"
path = "/cluster/home/nihars/mdeNeRF/newgroup/nihars_tests/kitti/datasets_cvpr/2011_09_26_drive_0001_sync_0/pointcloud.npz"

point_cloud_data = np.load(path)
point_cloud_points = point_cloud_data['xyz']
point_cloud_colors = point_cloud_data['rgb']

# load json file from path
with open(jsonfile) as f:
    json_data = json.load(f)

# get the list of frames
frames = json_data['frames']

# Get the cam intrinsics from json file from format:
# Extract the relevant values
fx = json_data["fl_x"]
fy = json_data["fl_y"]
cx = json_data["cx"]
cy = json_data["cy"]

H = json_data["h"]
W = json_data["w"]

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Extract the transform_matrix from the JSON data
transform_matrix = json_data["frames"][0]["transform_matrix"]

# Convert the transform_matrix to a NumPy array
extrinsic_matrix = np.array(transform_matrix)

extrinsic_matrix[0:3, 1:3] *= -1

# Rotate 180° and see road from other side
# extrinsic_matrix = np.dot(extrinsic_matrix, np.array([[1, 0, 0, 0], 
#                                                         [0, -1, 0, 7],
#                                                         [0, 0, 1, 0],
#                                                         [0, 0, 0, 1]]))


transformed_point_cloud = np.dot(np.linalg.inv(extrinsic_matrix)[:3, :3], point_cloud_points.T) + np.linalg.inv(extrinsic_matrix)[:3, 3][:, np.newaxis]
mask = transformed_point_cloud[2, :] > 0 & (transformed_point_cloud[2, :] < 50)

transformed_point_cloud = transformed_point_cloud[:, mask]
point_cloud_colors = point_cloud_colors[mask, :]

# sort the colors and points by the distance from the camera in z direction in descending order
idx = np.argsort(transformed_point_cloud[2, :])[::-1]
transformed_point_cloud = transformed_point_cloud[:, idx]
point_cloud_colors = point_cloud_colors[idx, :]


# Project the transformed points to 2D image coordinates
projected_points = np.dot(K, transformed_point_cloud)
projected_points /= projected_points[2, :] 

mask = (projected_points[0, :] >= 0) & (projected_points[0, :] < W) & (projected_points[1, :] >= 0) & (projected_points[1, :] < H)
projected_points = projected_points[:, mask].astype(np.int)
point_cloud_colors = point_cloud_colors[mask, :]

# breakpoint()

img = np.zeros((H, W, 3))
mask = np.zeros((H, W))

for point, color in zip(projected_points.T, point_cloud_colors):
    img[int(point[1]), int(point[0])] = color

for point in projected_points.T:
    mask[int(point[1]), int(point[0])] = 1

cv.imwrite("test.png", img)
cv.imwrite("mask.png", mask * 255)