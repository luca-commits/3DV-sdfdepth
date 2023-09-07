import os
import argparse
import json

import numpy as np
import pykitti

def main(args):
    date = args.scene[:10]
    drive = args.scene[-9:-5]

    calib_data = pykitti.raw(args.rgb_base_path, date, drive)

    # List to save transform matrix and paths for each frame
    frames = []

    cam_folders = []

    if "2" in args.cameras:
        cam_folders.append("image_02")
    if "3" in args.cameras:
        cam_folders.append("image_03")

    if len(cam_folders) == 0:
        raise RuntimeError("No cameras to process. Check --camera argument")

    # Iterate over both cameras
    for cam_folder in cam_folders:
        rgb_path = f"{args.rgb_base_path}/{date}/{args.scene}/{cam_folder}/data"
        depth_path = f"{args.depth_base_path}/{args.scene}/proj_depth/groundtruth/{cam_folder}"

        rgb_files = [ f for f in os.listdir(rgb_path) if ".png" in f ]
        depth_files = [ f for f in os.listdir(depth_path) if ".png" in f ]

        # Intersection of RGB and depth images
        filenames = [ f for f in sorted(rgb_files) if f in depth_files ]

        # Iterate over all frames captured by the current camera
        for filename in filenames:
            i = int(filename.replace(".png", ""))

            if i < int(args.first_index) or i >= int(args.last_index):
                continue

            if cam_folder == "image_02":
                calib_matrix = calib_data.calib.T_cam2_imu
            else:
                calib_matrix = calib_data.calib.T_cam3_imu

            # Adjust coordinate systems
            rotmat = np.transpose(np.array([[0,  1, 0, 0],
                                            [-1, 0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]]))

            transform_matrix = rotmat.dot(calib_data.oxts[i].T_w_imu.dot(np.linalg.inv(calib_matrix)))
            transform_matrix[0:3, 1:3] *= -1

            frames.append({
                "transform_matrix": transform_matrix.tolist(),
                "file_path": f"{rgb_path}/{filename}",
                "depth_file_path": f"{depth_path}/{filename}"
            })

    intrinsics = {
        "fl_x": calib_data.calib.P_rect_20[0, 0],
        "fl_y": calib_data.calib.P_rect_20[1, 1],
        "cx": calib_data.calib.P_rect_20[0, 2],
        "cy": calib_data.calib.P_rect_20[1, 2],
        "w": 1242,
        "h": 375,
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
    with open(os.path.join(args.save_path,'transforms.json'),'w') as f:
        json.dump(transforms, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw KITTI scence into a Nerfstudio dataset")
    parser.add_argument("--rgb-base-path", dest="rgb_base_path", required=True, help="Path to the RGB images of the raw dataset")
    parser.add_argument("--depth-base-path", dest="depth_base_path", required=True, help="Path to the depth images of the raw dataset")
    parser.add_argument("--scene", dest="scene", required=True)
    parser.add_argument("--save-path", dest="save_path", required=True, help="Where to drop the transforms.json file")
    parser.add_argument("--first_index", dest="first_index", required=False, default=0)
    parser.add_argument("--last_index", dest="last_index", required=False, default=100000)
    parser.add_argument("--camera", dest="cameras", nargs="+", default=["2", "3"])

    args = parser.parse_args()

    main(args)