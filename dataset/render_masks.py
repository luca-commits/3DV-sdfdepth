import numpy as np
import os 
import json
import cv2 as cv
np.set_printoptions(precision=2)

# No need to change this one. Camera model should be constant
jsonfile = "/cluster/home/nihars/mdeNeRF/newgroup/cfeldmann/for_nikolas/transforms.json"

# Change this one to the path where the renders are stored
poses_path = "/cluster/home/nihars/mdeNeRF/newgroup/nihars_tests/renders_angled_15/"

# load json file from path
with open(jsonfile) as f:
    json_data = json.load(f)

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

dirs = os.listdir(poses_path)

# scenes 
for dir in dirs:
    path = f"/cluster/home/nihars/mdeNeRF/newgroup/nihars_tests/kitti/datasets_cvpr/{dir}/pointcloud.npz"
    if not os.path.isfile(path):
        print(f"skipping {path}")
        continue

    poses_file_dirs = os.path.join(poses_path, dir)

    # load pc
    point_cloud_data = np.load(path)
    point_cloud_points = point_cloud_data['xyz']
    point_cloud_colors_full = point_cloud_data['rgb']

    # pos, neg angle
    for pose_file_dir in os.listdir(poses_file_dirs):
        poses_file = os.path.join(poses_file_dirs, pose_file_dir, "cam_poses.json")  
        render_path = os.path.join(poses_file_dirs, pose_file_dir)

        # if poses_file is a file
        if not os.path.isfile(poses_file):
            print(f"skipping {poses_file}")
            continue

        with open(poses_file) as f:
            frames = json.load(f)


        # ----------------- for loop should start here ----------------

        # Extract the transform_matrix from the JSON data
        for i, frame in enumerate(frames):
            #transform_matrix = frame["transform_matrix"]
            if os.path.isfile(os.path.join(render_path, f"{i:05d}_mask.png")):
                print(f"EXISTS: skipping {os.path.join(render_path, f'{i:05d}_mask.png')}")
                continue

            # Convert the transform_matrix to a NumPy array
            extrinsic_matrix = np.array(frame + [[0,0,0,1]])

            inv_extrinsic_matrix =  np.linalg.inv(extrinsic_matrix)

            transformed_point_cloud = np.dot(inv_extrinsic_matrix[:3, :3], point_cloud_points.T) + inv_extrinsic_matrix[:3, 3][:, np.newaxis]
            mask = transformed_point_cloud[2, :] > 0 & (transformed_point_cloud[2, :] < 80)
            transformed_point_cloud = transformed_point_cloud[:, mask]
            # point_cloud_colors = point_cloud_colors_full[mask, :]

            # Project the transformed points to 2D image coordinates
            projected_points = np.dot(K, transformed_point_cloud)
            projected_points /= projected_points[2, :] 

            mask = (projected_points[0, :] >= 0) & (projected_points[0, :] < W) & (projected_points[1, :] >= 0) & (projected_points[1, :] < H)
            projected_points = projected_points[:, mask].astype(int)
            # point_cloud_colors = point_cloud_colors[mask, :]

            mask = np.zeros((H, W))

            mask[projected_points[1], projected_points[0]] = 1

            # img = cv.imread(os.path.join(render_path, f"{i:05d}_rgb.png"))
            # for point, color in zip(projected_points.T, point_cloud_colors):
            #     img[int(point[1]), int(point[0])] = color

            # cv.imwrite(os.path.join(render_path, f"{i:05d}_real_rgb.png"), img)
            cv.imwrite(os.path.join(render_path, f"{i:05d}_mask.png"), mask * 255)
            # cv.imwrite(os.path.join(render_path, f"{i:05d}_rgb_gt.png"), img)
            print(f"saved {os.path.join(render_path, f'{i:05d}_mask.png')}")