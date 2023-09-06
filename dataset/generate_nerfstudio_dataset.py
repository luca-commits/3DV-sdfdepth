import os
import argparse
import json

import numpy as np
import pykitti

def main(args):
    # scene = "2011_09_26_drive_0001_sync"
    date = args.scene[:10]
    drive = args.scene[-9:-5]

    rotmat = np.transpose(np.array([[0,  1, 0, 0],
                                    [-1, 0, 0, 0],
                                    [0,  0, 1, 0],
                                    [0,  0, 0, 1]]))

    # rgb_base_path = f"/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images"
    # depth_base_path = f"/cluster/project/infk/courses/252-0579-00L/group26/kitti/depth/data_depth_annotated/train"
    # save_path = "/cluster/project/infk/courses/252-0579-00L/group26/sniall/3dv_sdfdepth/dataset"

    calib_data = pykitti.raw(args.rgb_base_path, date, drive)

    frames = []

    for cam_folder in ["image_02", "image_03"]:
        rgb_path = f"{args.rgb_base_path}/{date}/{args.scene}/{cam_folder}/data"
        depth_path = f"{args.depth_base_path}/{args.scene}/proj_depth/groundtruth/{cam_folder}"

        rgb_files = [ f for f in os.listdir(rgb_path) if ".png" in f ]
        depth_files = [ f for f in os.listdir(depth_path) if ".png" in f ]

        # Intersection of RGB and depth images
        filenames = [ f for f in sorted(rgb_files) if f in depth_files ]

        for filename in filenames:
            i = int(filename.replace(".png", ""))

            if cam_folder == "image_02":
                calib_matrix = calib_data.calib.T_cam2_imu
            else:
                calib_matrix = calib_data.calib.T_cam3_imu

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

    with open(os.path.join(args.save_path,'transforms.json'),'w') as f:
        json.dump(transforms, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw KITTI scence into a Nerfstudio dataset")
    parser.add_argument("--rgb-base-path", dest="rgb_base_path", required=True, help="Path to the RGB images of the raw dataset")
    parser.add_argument("--depth-base-path", dest="depth_base_path", required=True, help="Path to the depth images of the raw dataset")
    parser.add_argument("--scene", dest="scene", required=True)
    parser.add_argument("--save-path", dest="save_path", required=True, help="Where to drop the transforms.json file")

    args = parser.parse_args()

    main(args)