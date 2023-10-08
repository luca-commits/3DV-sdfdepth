import numpy as np
import os 
import json
import cv2 as cv

transforms = "/cluster/home/nihars/mdeNeRF/newgroup/cfeldmann/for_nikolas/dataparser_transforms.json"
jsonfile = "/cluster/home/nihars/mdeNeRF/newgroup/cfeldmann/for_nikolas/transforms.json"
path = "/cluster/home/nihars/mdeNeRF/newgroup/nihars_tests/kitti/datasets_cvpr/2011_09_26_drive_0001_sync_0/pointcloud.npz"

# renders are here Naming: 00000_rgb.png
render_path = "/cluster/home/nihars/mdeNeRF/newgroup/cfeldmann/for_nikolas/renders"

point_cloud_data = np.load(path)
point_cloud_points = point_cloud_data['xyz']
point_cloud_colors_full = point_cloud_data['rgb']

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


# ----------------- for loop should start here ----------------

# Extract the transform_matrix from the JSON data
for i, frame in enumerate(frames):
    transform_matrix = frame["transform_matrix"]

    # Convert the transform_matrix to a NumPy array
    extrinsic_matrix = np.array(transform_matrix)

    extrinsic_matrix[0:3, 1:3] *= -1

    inv_extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

    transformed_point_cloud = np.dot(inv_extrinsic_matrix[:3, :3], point_cloud_points.T) + inv_extrinsic_matrix[:3, 3][:, np.newaxis]
    mask = transformed_point_cloud[2, :] > 0 & (transformed_point_cloud[2, :] < 80)
    transformed_point_cloud = transformed_point_cloud[:, mask]

    # Project the transformed points to 2D image coordinates
    projected_points = np.dot(K, transformed_point_cloud)
    projected_points /= projected_points[2, :] 

    mask = (projected_points[0, :] >= 0) & (projected_points[0, :] < W) & (projected_points[1, :] >= 0) & (projected_points[1, :] < H)
    projected_points = projected_points[:, mask].astype(int)

    mask = np.zeros((H, W))

    mask[projected_points[1], projected_points[0]] = 1

    # cv.imwrite(os.path.join(render_path, f"{i:05d}_real_rgb.png"), img)
    cv.imwrite(os.path.join(render_path, f"{i:05d}_mask.png"), mask * 255)
    print(f"saved {i:05d}.png")