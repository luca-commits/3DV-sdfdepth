# Deep Learning for Autonomous Driving
# Material for the 3rd and 4th Problems of Project 1
# For further questions contact Dengxin Dai, dai@vision.ee.ethz.ch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import dirname, abspath
import argparse
import math as m


def load_from_bin(bin_path):
    # load point cloud from a binary file
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:, :3]


def depth_color(val, min_d=0, max_d=100):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val)  # max distance is 120m but usually not usual
    # 150 to get a bigger range from blue to pink, but not return to red again
    return (((val - min_d) / (max_d - min_d)) * 150).astype(np.uint8)


def line_color(val, min_d=1, max_d=64):
    """
    print Color(HSV's H value) corresponding to laser id
    """
    alter_num = 4
    return (((val - min_d)%alter_num) * 127/alter_num).astype(np.uint8)




def calib_velo2cam(filepath):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T


# def calib_cam2cam(filepath, mode):
#     """
#     If your image is 'rectified image' :
#         get only Projection(P : 3x4) matrix is enough
#     but if your image is 'distorted image'(not rectified image) :
#         you need undistortion step using distortion coefficients(5 : D)
#
#     in this code, we'll get P matrix since we're using rectified image.
#     in this code, we set filepath = 'yourpath/2011_09_26_drive_0029_sync/calib_cam_to_cam.txt' and mode = '02'
#     """
#     with open(filepath, "r") as f:
#         file = f.readlines()
#
#         for line in file:
#             (key, val) = line.split(':', 1)
#             if key == ('P_rect_' + mode):
#                 P_ = np.fromstring(val, sep=' ')
#                 P_ = P_.reshape(3, 4)
#                 # erase 4th column ([0,0,0])
#                 P_ = P_[:3, :3]
#     return P_



def calib_cam2cam(filepath, mode):
    """
    projection matrix from reference camera coordinates to a point on the ith camera plan
    in our case, mode is '02'
    the rectifying rotation matrix "R_rect_00" of the referece camera needs to be considered for accurate projection
    lidar to cam2: first project lidar pts to cam0 by the matrices returned by calib_velo2cam, and
    then transform the projected pts with the matrix returned by this function
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                #P_ = P_[:3, :3]
            if key == ('R_rect_' + mode):
                R_rect_00 = np.fromstring(val, sep=' ')
                R_rect_00 = R_rect_00.reshape(3, 3)
                R_rect_00_ = np.zeros((4,4))
                R_rect_00_[:-1,:-1] = R_rect_00

    return np.matmul(P_, R_rect_00_)



def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Sort points by distance (color)
    idx = np.argsort(color)

    for i in idx[::-1]:
        # Start at 20 to have blue as close colour and red as far colour
        col = 180 - int(color[i]) if color[i] <= 180 else 180 - int(color[i])
        hsv_image[np.int32(points[1][i]), np.int32(points[0][i])] = np.array((col, 255, 255))

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def gray_to_dist(depth, image):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # breakpoint()
    if depth.max() > 255:
        depth = (depth / 256).astype(np.uint8)
    else:
        depth = depth.astype(np.uint8)


    # Sort points by distance (color)
    xs = image.shape[0]
    ys = image.shape[1]
    # breakpoint()
    for x in range(xs):
        for y in range(ys):
            if depth[x,y] == 0:
                continue
            # breakpoint()
            # Start at 20 to have blue as close colour and red as far colour
            col = 180 - int(depth[x,y]) if depth[x,y] <= 180 else 180 - int(depth[x,y])
            hsv_image[x, y] = np.array((col, 255, 255))

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def print_projection_dists(points, dists, image, empty_image: bool = True):
    """ project converted velodyne points into camera image """
    if empty_image:
        depth_map = np.zeros((image.shape[1], image.shape[0]))
    else:
        depth_map = image

    # Sort points by distance (color)
    idx = np.argsort(dists)

    for i in idx[::-1]:
        depth_map[np.int32(points[0][i]), np.int32(points[1][i])] = dists[i]

    return depth_map



def compute_timestamps(timestamps_f, ind):
    # return timestamps of the the ind^th sample (line) in seconds
    # in this code, timestamps_f can be 'image_02/timestamps.txt', 'oxts/timestamps.txt', 'velodyne_points/timestamps_start.txt', ...
    #  'velodyne_points/timestamps_end.txt',  or 'velodyne_points/timestamps.txt'. ind is the index (name) of the sample like '0000000003'
    with open(timestamps_f) as f:
        timestamps_ = f.readlines()
        #file_id = file[7:10]
        timestamps_ = timestamps_[int(ind)]
        timestamps_ = timestamps_[11:]
        timestamps_ = np.double(timestamps_[:2]) * 3600 + np.double(timestamps_[3:5]) * 60 + np.double(timestamps_[6:])
    return timestamps_


def load_oxts_velocity(oxts_f):
    # return the speed of the vehicle given the oxts file
    with open(oxts_f) as f:
        data = [list(map(np.double, line.strip().split(' '))) for line in f]
        speed_f = data[0][8]
        speed_l = data[0][9]
        speed_u = data[0][10]
    return np.array((speed_f, speed_l, speed_u))


def load_oxts_angular_rate(oxts_f):
    # return the angular rate of the vehicle given the oxts file
    with open(oxts_f) as f:
        data = [list(map(np.double, line.strip().split(' '))) for line in f]
        angular_rate_f = data[0][20]
        angular_rate_l = data[0][21]
        angular_rate_u = data[0][22]
    return angular_rate_f, angular_rate_l, angular_rate_u