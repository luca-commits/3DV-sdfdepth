import os
import argparse
import json

import numpy as np
import pykitti
import cv2
import shutil

from projections_inv import SensorFusionInv

def main(args):
    date = args.scene[:10]
    drive = args.scene[-9:-5]

    # Corresponds to 'data'
    calib_data = pykitti.raw(args.rgb_base_path, date, drive)

    base_data_folder = f'{args.rgb_base_path}/{date}'
    cam_calibs = base_data_folder + '/calib_cam_to_cam.txt'

    data_folder = os.path.join(base_data_folder, f"{date}_drive_{drive}_sync")
    save_loc = f"/cluster/project/infk/courses/252-0579-00L/group26/nihars_tests/kitti/{date}_drive_{drive}_sync/"
    annotated_depths_path = os.path.join(f"{date}_drive_{drive}_sync/proj_depth/groundtruth/")


    cam_folders = []

    if "2" in args.cameras:
        cam_folders.append("image_02")
    if "3" in args.cameras:
        cam_folders.append("image_03")

    if len(cam_folders) == 0:
        raise RuntimeError("No cameras to process. Check --camera argument")
    

    #Now we gather the (x,y,z) poses of the cameras
    pose_list = []

    # Need the position of one of the cameras
    if "image_02" in cam_folders:
        cam_folder = "image_02"
    else:
        cam_folder = "image_03"

    rgb_path = f"{args.rgb_base_path}/{date}/{args.scene}/{cam_folder}/data"
    depth_path = f"{args.depth_base_path}/{args.scene}/proj_depth/groundtruth/{cam_folder}"

    rgb_files = [ f for f in os.listdir(rgb_path) if ".png" in f ]
    depth_files = [ f for f in os.listdir(depth_path) if ".png" in f ]

    # Intersection of RGB and depth images
    filenames = [ f for f in sorted(rgb_files) if f in depth_files ]

    # Iterate over all frames captured by the current camera
    for filename in filenames:
        i = int(filename.replace(".png", ""))

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

        pose_list.append(transform_matrix[0:3,3])

    poses = np.vstack(pose_list)
    pose_mean = np.mean(poses, axis=0)
    centered_poses = poses - pose_mean
    # scale = np.max(np.abs(centered_poses))

    scene_cuts = [0]

    #Once the pose is too far away from the beginning pose, cut and repeat
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

        # print("Scene:", k)

        # List to save transform matrix and paths for each frame
        frames = []

        fused_sensors = SensorFusionInv(base_data_folder, calib_data)
        xyz_pointcloud = []
        rgb_pointcloud = []
        # Iterate over both cameras
        for cam_folder in cam_folders:
            rgb_path = f"{args.rgb_base_path}/{date}/{args.scene}/{cam_folder}/data"
            depth_path = f"{args.depth_base_path}/{args.scene}/proj_depth/groundtruth/{cam_folder}"

            rgb_files = [ f for f in os.listdir(rgb_path) if ".png" in f ]
            depth_files = [ f for f in os.listdir(depth_path) if ".png" in f ]

            # Intersection of RGB and depth images
            # Only add the images which belong to the section between the current and next cut
            filenames = [ f for f in sorted(rgb_files) if (f in depth_files and check_if_valid_file(f, scene_cuts, k)) ]

            cam_id = int(cam_folder[-2:])
            # breakpoint()
            if cam_folder == "image_02":
                calib_matrix = calib_data.calib.T_cam2_imu
            else:
                calib_matrix = calib_data.calib.T_cam3_imu
            # Adjust coordinate systems
            rotmat = np.transpose(np.array([[0,  1, 0, 0],
                                            [-1, 0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]]))
        
            filenames = [ f for f in sorted(filenames) if check_if_valid_velocity(f, calib_data)]
            transform_matrices = [get_transform_mat(rotmat, calib_data, calib_matrix, f) for f in sorted(filenames)]
            transform_matrices_pc = [get_transform_mat_pc(rotmat, calib_data, f) for f in sorted(filenames)]

            # For mesh
            frame_ids = [int(f.replace(".png", "")) for f in sorted(filenames)]
            depth_img_paths = [os.path.join(depth_path, f) for f in sorted(filenames)]
            rgb_img_paths = [os.path.join(rgb_path, f) for f in sorted(filenames)]

            # frame_ids: List[int] = None, depth_images: List[np.ndarray] = None, rgb_images: List[np.ndarray] = None)
            pc_tmp, rgb_tmp = fused_sensors(frame_ids, depth_img_paths, rgb_img_paths, transform_matrices_pc, cam=cam_id) # Transformation from world to imu
            xyz_pointcloud.append(pc_tmp)
            rgb_pointcloud.append(rgb_tmp)

            # Iterate over all valid frames captured by the current camera
            for filename, transform_matrix in zip(filenames, transform_matrices):
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
        merged_pc = np.concatenate(xyz_pointcloud)
        merged_rgb = np.concatenate(rgb_pointcloud)
        np.savez(os.path.join(indexed_save_path, "pointcloud.npz"), xyz=merged_pc, rgb=merged_rgb)


        intrinsics = {
            "fl_x": calib_data.calib.P_rect_20[0, 0],
            "fl_y": calib_data.calib.P_rect_20[1, 1],
            "cx": calib_data.calib.P_rect_20[0, 2],
            "cy": calib_data.calib.P_rect_20[1, 2],
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
        with open(os.path.join(indexed_save_path, 'transforms.json'),'w') as f:
            json.dump(transforms, f, indent=4)


def check_if_valid_file(filename, scene_cuts, k):
    i = int(filename.replace(".png", ""))
    return (i >= scene_cuts[k] and (k == len(scene_cuts)-1 or i < scene_cuts[k+1]))

def check_if_valid_velocity(filename, calib_data, thr=1.0):
    i = int(filename.replace(".png", ""))
    return calib_data.oxts[i].packet.vf >= thr

def get_transform_mat(rotmat, calib_data, calib_matrix, filename):
    i = int(filename.replace(".png", ""))
    transform_matrix = rotmat.dot(calib_data.oxts[i].T_w_imu.dot(np.linalg.inv(calib_matrix)))
    transform_matrix[0:3, 1:3] *= -1
    return transform_matrix

def get_transform_mat_pc(rotmat, calib_data, filename):
    velo_2_imu = calib_data.calib.T_velo_imu
    i = int(filename.replace(".png", ""))
    transform_matrix = rotmat.dot(calib_data.oxts[i].T_w_imu.dot(velo_2_imu))
    transform_matrix[0:3, 1:3] *= -1
    return transform_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw KITTI scence into a Nerfstudio dataset")
    parser.add_argument("--rgb-base-path", dest="rgb_base_path", required=True, help="Path to the RGB images of the raw dataset")
    parser.add_argument("--depth-base-path", dest="depth_base_path", required=True, help="Path to the depth images of the raw dataset")
    parser.add_argument("--scene", dest="scene", required=True)
    parser.add_argument("--save-path", dest="save_path", required=True, help="Where to drop the transforms.json file")
    parser.add_argument("--camera", dest="cameras", nargs="+", default=["2", "3"])
    parser.add_argument("--max-scene-size", dest="max_scene_size", default=50.0, type=float)

    args = parser.parse_args()

    main(args)