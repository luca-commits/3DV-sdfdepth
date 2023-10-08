# Task 2: Sensor Fusion + Untwisting of data

# import packages
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Any, List, Union
# import helper function
from data_utils import *


from typing import Tuple, Union
import os
import cv2 as cv
import cv2
import sys

# Deep Learning for Autonomous Driving
# Material for Problems 1-3 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import pickle

def load_data(data_path):
    ''' 
    Load data dictionary from data_path.
    '''
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

class Sensordata:
    """
    Generic container class for any sensor data.
    Is basically Velodyne
    """
    def __init__(self, frame_id: int = None):
        """ Should be faster with frame id """
        self.data = []
        self.cat_dir = "None"
        self.dir_name = "velodyne_points"
        self.flag = None
        self.load_function = load_from_bin

    def __getitem__(self, item: int) -> np.ndarray:
        """ Returns data of frame with id == item"""
        return self.data[item]

    def __setitem__(self, key, value):
        """ Sets the data of frame: key to the wanted value: value"""
        self.data[key] = value

    def __call__(self,
                 id: int = 0,
                 get_start_end: bool = False
                 ) -> Union[Tuple[np.float64], Tuple[np.float64, np.float64, np.float64]]:
        """ Returns item and also timestamps. If velodyne then it can also return the start and end timestamps"""
        ts = compute_timestamps(os.path.join(self.cat_dir, "timestamps.txt"), id)
        return ts

    def load_data(self, need_data_dir = True) -> None:
        """
        Loads data into self.data as a list
        """
        if need_data_dir:
            data_path = "data"
            self.cat_dir = self.dir_name
            data_dir = os.path.join(self.cat_dir, data_path)
        else:
            data_dir = self.dir_name
        listed_files = sorted([file for file in os.listdir(data_dir) if not file.endswith(".ipynb_checkpoints")])
        for file in listed_files:
            if self.flag is not None:
                self.data.append(self.load_function(os.path.join(data_dir, file), self.flag))
            else:
                self.data.append(self.load_function(os.path.join(data_dir, file)))
    

class SensorFusionInv:
    """
    Holds all sensors in one interface class
    """
    def __init__(self,
                 directory: str, 
                 data):

        # Sensors
        self.data_kitti = data

        self.problem_dir = directory
        self.get_calibrations_matrices(2)


    def get_calibrations_matrices(self, camera_id: int = None):
        """ Get Matrices from the files and process them"""
        if camera_id is None:
            raise ValueError("Please specify camera id")
        print(f"Loaded camera calibs of Camera {camera_id}")
        if camera_id == 2:
            self.K_c2c = np.concatenate([np.array(self.data_kitti.calib.K_cam2), np.array([[0,0,0]]).T], 1) #calib_cam2cam(os.path.join(self.problem_dir, "calib_cam_to_cam.txt"), mode=f"0{self.camera_id}")
            self.Rt_v2c = np.array(self.data_kitti.calib.T_cam2_velo)# 
            self.test_Rt_v2c = get_hom_Rt_matrix(*calib_velo2cam(os.path.join(self.problem_dir, "calib_velo_to_cam.txt")))
        elif camera_id == 3:
            self.K_c2c = np.concatenate([np.array(self.data_kitti.calib.K_cam3), np.array([[0,0,0]]).T], 1)
            self.Rt_v2c = np.array(self.data_kitti.calib.T_cam3_velo)# get_hom_Rt_matrix(*calib_velo2cam(os.path.join(self.problem_dir, "calib_velo_to_cam.txt")))
        self.Rt_imu2v_h = get_hom_Rt_matrix(*calib_velo2cam(os.path.join(self.problem_dir, "calib_imu_to_velo.txt")))
        self.K_c2c_inv = np.linalg.inv(np.concatenate([self.K_c2c, np.array([[0,0,0,1]])], 0))
        # breakpoint()

    # frame_ids, depth_img_paths, rgb_img_paths, transform_matrices
    def __call__(self, frame_ids: List[int] = None, depth_images: List[np.ndarray] = None, rgb_images: List[np.ndarray] = None, extrinsics = None, cam = None) -> np.ndarray:
        self.start_frame_idx = frame_ids[0]
        init_frame_id = frame_ids[0]
        init_rgb = rgb_images[0]
        init_depth = depth_images[0]
        init_extrinsics = extrinsics[0] 

        # breakpoint()
        points = []
        colors = []
        self.get_calibrations_matrices(cam)

        # breakpoint()


        # Get image and points from initial frame first
        tmp_points, tmp_cols = self.process_frame_inv(init_frame_id, init_frame_id, init_rgb, init_depth, init_extrinsics, cam=cam)
        points.append(tmp_points)
        colors.append(tmp_cols)

        # Get image and points from the following frames
        for frame_id, depth_img, rgb_img, extr in zip(frame_ids[1:], depth_images[1:], rgb_images[1:], extrinsics[1:]):
            try:
                tmp_points, tmp_cols = self.process_frame_inv(init_frame_id, frame_id, rgb_img, depth_img, extr, cam=cam)
                points.append(tmp_points)
                colors.append(tmp_cols)
                print(f"Cam {cam}: Frame {frame_id}")
            except ValueError:
                break
        
        # Making the depth image. Reversing the order s.t. closer frames overwrite further frames 
        return np.concatenate(points), np.concatenate(colors)

    # init_frame_id, init_frame_id, init_rgb, init_depth, init_extrinsics, cam=cam)
    def process_frame_inv(self, init_frame_id: int, frame_id: int, rgb_image = None, depth_image = None, extr = None, cam=2):
        """
        Sorts the PC first based on their Yaw angle (angle around Z-axis) and then
            - computes the rotation and translation that happened between the start - trigger - end
            - plots this
        """
        try:
            # Check if frame_id is in range otherwise raise error which is caught by __call__
            self.data_kitti.oxts[frame_id]
            self.data_kitti.oxts[init_frame_id]
        except:
            raise ValueError("Frame ID out of range")
        
        # Depth image as depth values per pixel
        depth_img_pixels = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED) / 256.0
        mask = depth_img_pixels != 0

        # breakpoint()
        img = cv2.cvtColor(cv.imread(rgb_image), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        img = img[mask.reshape(-1)]

        # Depth image as point cloud in velodyne frame
        depth_img_v_frame = self.depth_image_to_point_cloud(depth_img_pixels, self.K_c2c, np.linalg.inv(self.Rt_v2c), mask=mask)

        #self.imu2w = self.data_kitti.oxts[init_frame_id].T_w_imu # imu2w
        #self.T_rel_wnew2w = self.data_kitti.oxts[frame_id].T_w_imu # @ self.imu2w # Trafo in reference coords 
        # breakpoint()
        self.T_rel_wnew2w = extr #self.T_rel_wnew2w @ np.linalg.inv(self.Rt_imu2v_h) #extr
        self.T_rel_wnew2w[0:3, 1:3] *= -1
        velos_imu = transform(self.T_rel_wnew2w, depth_img_v_frame)
        velos_imu = (velos_imu[:3, :]/ velos_imu[3, :]).T

        # if init_frame_id != frame_id:
        #     breakpoint()
        
        # velos_trafoed_imu = transform(np.linalg.inv(self.T_rel_wnew2w), velos_imu)
        # velos_trafoed_imu = (velos_trafoed_imu[:3, :] / velos_trafoed_imu[3, :]).T

        # breakpoint()
        return velos_imu, img


    def depth_image_to_point_cloud(self, depth_image, intrinsics, extrinsics, mask):
        # Get the shape of the depth image
        H, W = depth_image.shape

        # Create pixel coordinates grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Flatten the pixel coordinates and depth values
        u_flat = u[mask].flatten()
        v_flat = v[mask].flatten() 
        depth_flat = depth_image[mask].flatten()

        # Apply camera intrinsics to get normalized image coordinates
        u_normalized = (u_flat - intrinsics[0, 2]) / intrinsics[0, 0]
        v_normalized = (v_flat - intrinsics[1, 2]) / intrinsics[1, 1]

        # Calculate 3D camera coordinates (X_c, Y_c, Z_c)
        X_c = u_normalized * depth_flat
        Y_c = v_normalized * depth_flat
        Z_c = depth_flat

        # Create homogeneous coordinates
        ones = np.ones_like(X_c)

        # Combine X_c, Y_c, Z_c, and ones into a homogeneous point cloud
        point_cloud = np.vstack((X_c, Y_c, Z_c, ones))

        # Apply camera extrinsics to transform to world coordinates
        point_cloud_world = np.dot(np.linalg.inv(self.Rt_v2c), point_cloud)

        # Transpose to have each column represent a 3D point
        point_cloud_world = (point_cloud_world[:3] / point_cloud_world[3]).T

        return point_cloud_world


def transform(M: np.ndarray, pc: np.ndarray, idx: np.ndarray = None, dim: int = 3) -> np.ndarray:
    """
    Returns M * | PC |
                | 1  |
    idx are the filtered indices
    """
    if len(np.shape(M)) == 3:
        return np.array([np.matmul(m, p) for m, p in zip(M, pc.T)])
    else:
        if idx is None:
            return np.matmul(M, np.c_[pc[:, :dim], np.ones(pc.shape[0])].T)
        else:
            return np.matmul(M, np.c_[pc[idx][:, :dim], np.ones(len(idx))].T)


def get_hom_Rt_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Takes Rotation and Translation Matrix and Returns:
        | R  t |
    M = | 0  1 |

    """
    Rt = np.concatenate([R, t], axis=1)
    dim_2 = Rt.shape[1]
    hom = np.zeros((1, dim_2)); hom[0, dim_2-1] = 1
    return np.concatenate([Rt, hom])


def cartesian_to_polar(pc: np.ndarray, ret_xy_dist: bool = True) -> Tuple[np.array, np.array]:
    """ Converts cartesian to polar/ "spherical" coordinates.
    :param pc: Pointcloud
    :param ret_xy_dist: bool - True  -> returns polar coordinate in the xy plane
                             - False -> returns azimuth angle and radius from spherical coordinates
    """
    dim = 3 - int(ret_xy_dist)
    dist = np.linalg.norm(pc[:, :dim].copy(), axis=1)
    angle = -np.rad2deg(np.arctan2(pc[:, 1], pc[:, 0]))
    return dist, angle


def rot_and_trans(t: np.array, w: np.float64, vx: np.float64, vy: np.float64, pc: np.ndarray) -> np.ndarray:
    """
    Rotate and Translate with vectors:
    new_x = x * cos(a) - y * sin(a) + dx
    new_x = x * sin(a) + y * cos(a) + dy
    new_z = z
    """
    cos_wt = np.cos(t*w)
    sin_wt = np.sin(t*w)
    x_cos = cos_wt * pc[:, 0]
    x_sin = sin_wt * pc[:, 0]

    y_sin = sin_wt * pc[:, 1]
    y_cos = cos_wt * pc[:, 1]

    new_x = x_cos - y_sin + vx * t
    new_y = x_sin + y_cos + vy * t
    return np.stack([new_x, new_y, pc[:, 2]]).T


def rotate(t: np.array, w: np.float64, pc: np.ndarray) -> np.ndarray:
    """
    Rotate with vectors:
    new_x = x * cos(a) - y * sin(a)
    new_x = x * sin(a) + y * cos(a)
    new_z = z

        | cos(a)   -sin(a)   0 |
    R = | sin(a)    cos(a)   0 |
        |   0         0      1 |
    """
    cos_wt = np.cos(t*w)
    sin_wt = np.sin(t*w)
    x_cos = cos_wt * pc[:, 0]
    x_sin = sin_wt * pc[:, 0]

    y_sin = sin_wt * pc[:, 1]
    y_cos = cos_wt * pc[:, 1]

    new_x = x_cos - y_sin
    new_y = x_sin + y_cos
    return np.stack([new_x, new_y, pc[:, 2]]).T


def translate(t: np.array, vx: np.float64, vy: np.float64, pc: np.ndarray) -> np.ndarray:
    """
    Translate with vectors:
    new_x = x + dx
    new_y = y + dy
    new_z = z
    """
    dx = vx * t
    dy = vy * t
    new_x = pc[:, 0] + dx
    new_y = pc[:, 1] + dy
    return np.stack([new_x, new_y, pc[:, 2]]).T


if __name__ == "__main__":
    frame = [3, 37, 77, 310]

    if isinstance(frame, int):
        sensors = SensorFusionInv(frame_id=frame, plot=True, imsave=True, waitkey=2000)
        sensors.process_and_plot_frame()
        sensors.pc_plotter(twisted_save=True)
    if isinstance(frame, list):
        for frame_id in frame:
            sensors = SensorFusionInv(frame_id=frame_id, plot=True, imsave=True, waitkey=2000)
            sensors.process_and_plot_frame()
            sensors.pc_plotter(twisted_save=True)
            cv.destroyAllWindows()
    else:
        for i in range(420):
            sensors = SensorFusionInv(frame_id=i, plot=True, imsave=False)
            sensors.process_and_plot_frame()
    cv.destroyAllWindows()