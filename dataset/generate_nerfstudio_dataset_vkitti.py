import os
import argparse
import json

import numpy as np
import cv2
import pandas as pd
import shutil

def main(args):
    # List to save transform matrix and paths for each frame
    frames = []

    rgb_path = os.path.join(args.rgb_base_path, args.scene, args.world)
    depth_path = os.path.join(args.depth_base_path, args.scene, args.world)
    extrinsic_file = os.path.join(args.extrinsic_base_path, f"{args.scene}_{args.world}.txt")

    rgb_files = [ f for f in os.listdir(rgb_path) if ".png" in f ]
    depth_files = [ f for f in os.listdir(depth_path) if ".png" in f ]

    # Intersection of RGB and depth images
    filenames = [ f for f in sorted(rgb_files) if f in depth_files ]

    # Load extrinsic calibration
    extgt = pd.read_csv(extrinsic_file, sep=" ", index_col=False)

    pose_list = []
    
    # Iterate over all frames captured by the current camera
    for filename in filenames:
        i = int(filename.replace(".png", ""))

        ext_matrix = np.array(extgt.iloc[i, 1:17]).reshape(4, 4)

        transform_matrix = np.linalg.inv(ext_matrix)
        transform_matrix[0:3, 1:3] *= -1
        transform_matrix = transform_matrix[np.array([1, 0, 2, 3]), :]
        transform_matrix[2, :] *= -1

        pose_list.append(transform_matrix[0:3,3])

    poses = np.vstack(pose_list)
    pose_mean = np.mean(poses, axis=0)
    centered_poses = poses - pose_mean

    scene_cuts = [0]

    start = centered_poses[0,:]
    for i in range(centered_poses.shape[0]):
        curr = centered_poses[i,:]
        if np.linalg.norm(curr - start) > args.max_scene_size:
            scene_cuts.append(i-1)
            start = curr

    print(scene_cuts)

    for k in range(len(scene_cuts)):

        indexed_save_path = args.save_path + "_" + str(k)

        os.makedirs(indexed_save_path, exist_ok=True)

        frames = []

        for filename in filenames:
            i = int(filename.replace(".png", ""))

            if i >= scene_cuts[k] and (k == len(scene_cuts)-1 or i < scene_cuts[k+1]):
                ext_matrix = np.array(extgt.iloc[i, 1:17]).reshape(4, 4)

                transform_matrix = np.linalg.inv(ext_matrix)
                transform_matrix[0:3, 1:3] *= -1
                transform_matrix = transform_matrix[np.array([1, 0, 2, 3]), :]
                transform_matrix[2, :] *= -1

                frames.append({
                    "transform_matrix": transform_matrix.tolist(),
                    "file_path": f"{rgb_path}/{filename}",
                    "depth_file_path": f"{depth_path}/{filename}"
                })
    
        if len(frames) > 0:
            height, width, _ = cv2.imread(frames[0]["file_path"]).shape
        else:
            print("WARNING: Dataset is empty")
            height, width = 1, 1

            # Delete created empty directory
            try:
                shutil.rmtree(indexed_save_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            # Skip saving of transform.json
            continue

        intrinsics = {
            "fl_x": 725,
            "fl_y": 725,
            "cx": 620.5,
            "cy": 187.0,
            "w": width,
            "h": height,
            "camera_model": "OPENCV",
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0
        }

        transforms = {
            **intrinsics,
            "frames": frames
        }

        # Save transforms
        with open(os.path.join(indexed_save_path,'transforms.json'),'w') as f:
            json.dump(transforms, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw KITTI scence into a Nerfstudio dataset")
    parser.add_argument("--rgb-base-path", dest="rgb_base_path", required=True, help="Path to the RGB images of the raw dataset")
    parser.add_argument("--depth-base-path", dest="depth_base_path", required=True, help="Path to the depth images of the raw dataset")
    parser.add_argument("--extrinsic-base-path", dest="extrinsic_base_path", required=True, help="Path to the extrinsic calibration files")
    parser.add_argument("--scene", dest="scene", required=True)
    parser.add_argument("--world", dest="world", required=False, default="clone")
    parser.add_argument("--save-path", dest="save_path", required=True, help="Where to drop the transforms.json file")
    parser.add_argument("--max-scene-size", dest="max_scene_size", default=50.0, type=float)

    args = parser.parse_args()

    main(args)