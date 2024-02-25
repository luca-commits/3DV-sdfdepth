import numpy as np
import cv2 as cv
import os
import argparse


class LidarSimulator:
    def __init__(self):
        # these are velodyne specific from the KITTI dataset (Velodyne HDL-64E)
        self.positive_vertical_fov = 2.0  # degrees
        self.negative_vertical_fov = -24.9  # degrees
        self.num_channels = 64

        self.channels = np.array(range(self.num_channels))
        self.max_dist = 120  # meters

        # equals roughly 0.42 degrees (0.4 is in datasheet) 
        self.ang_resolution = (self.positive_vertical_fov - self.negative_vertical_fov) / self.num_channels

        # self-defined
        self.horizontal_fov = 120.0
        self.height = 1.65

        # LIDAR captures about 100k points per 360° frame, each frame is 0.1s
        # so for 120° and per frame we have 100k * 120 / 360 = 33.3k points
        # this means per channel we have 33.3k / 64 = 520 points
        # means per degree we have 520 / 120 = 4.3 points
        self.points_per_frame = int(100e3 * self.horizontal_fov / 360.0)
        self.points_per_channel = int(self.points_per_frame / self.num_channels)
        self.points_per_degree = self.horizontal_fov / self.points_per_channel # approx 0.23 ° angle resolution (similar to datasheet)

        # do this once, then apply occupancy and add little measurement noise at every turn
        self.points = self.generate_rays()  # (N, 3) 

        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(self.points)
        # o3d.visualization.draw_geometries([point_cloud])  

    def generate_rays(self) -> np.ndarray:
        # assume that we are recording from 2m height and the whole ground around is perfectly flat
        # we can assume that the ground is at 0m and the maximum height is 2m
        # we can assume that the maximum distance is 120m
        
        yaws = np.array([np.pi / 180.0 * (-self.horizontal_fov / 2 + self.points_per_degree * point_idx) for point_idx in range(self.points_per_channel)])

        dividing_factors = np.abs(np.tan(np.pi / 180.0 * (self.negative_vertical_fov + self.channels * self.ang_resolution - self.positive_vertical_fov - 1e-6)))

        # divide-by-zero-handling
        distances = np.where(dividing_factors == 0, 120, self.height / dividing_factors)
        h_components = np.where(dividing_factors == 0, 0, self.height)


        X = distances[:, None] * np.sin(yaws[None, :])
        Z = distances[:, None] * np.cos(yaws[None, :])
        Y = h_components[:, None] * np.ones_like(yaws)[None, :]

        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        point_cloud = np.vstack((X, Y, Z)).T
        return point_cloud
    
    def get_noisy_points(self, occupancy = float, add_noise: bool = True):
        # copy the points and modify
        out_points = self.points.copy()
        
        # add measurement noise (~1cm inaccuracy in the Velodyne, ours will give less than that.)
        if add_noise:
            noise = (np.random.rand(*self.points.shape) - 0.5) / 170.0
            out_points = out_points + noise

        # The following generates the mask for specific random points 
        random_sparsification_noise = np.random.rand(self.points.shape[0])
        mask = random_sparsification_noise > (1 - occupancy)

        # mask out the points that we want to get rid of
        out_points = out_points[mask]
        return out_points
    
    def project_into_image_and_draw_dist(self, depth_img: np.ndarray, pc: np.ndarray, intr: np.ndarray):
        # project the point cloud into the image plane
        # pc: point cloud in the camera coordinate system
        # intr: intrinsic matrix
        # returns: 2D point cloud in the image plane
        pc_in_cam = np.dot(intr, pc.T).T
        pc_in_cam = (pc_in_cam[:, :2] / pc_in_cam[:, 2].reshape(-1, 1)).astype(np.int32)

        mask = (pc_in_cam[:, 0] >= 0) & (pc_in_cam[:, 0] < 1242) & (pc_in_cam[:, 1] >= 0) & (pc_in_cam[:, 1] < 375)
        pc_in_cam = pc_in_cam[mask]

        img_mask = np.zeros((375, 1242), dtype=bool)
        img_mask[pc_in_cam[:, 1], pc_in_cam[:, 0]] = True

        depth_img[img_mask == 0] = 0

        return depth_img


def main():
    # KITTI intrinsic matrix - maybe not 100% correct
    intr = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                    [0.000000e+00, 7.215377e+02, 1.728540e+02],
                    [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    
    # Generate it once and then vary the occupancy level for each image and occupancy level
    lidar = LidarSimulator()

    # this should be the path to the vkitti image
    path = "/home/nikolas/Documents/ETH/FS23/3DVision/NEWPROJ/supp_out/00000_depth.png"
    
    # these are (0.25, 0.5, 0.75, 1.0) (of the pseudo-LIDAR)
    occupancy_levels = np.linspace(0.25, 1.0, 4)
    for occupancy_level in occupancy_levels:
        # read depth image
        depth = cv.imread(path, cv.IMREAD_UNCHANGED)
        
        # add little measurement noise to the pseudo-LiDAR and filter out points for sparsification
        point_cloud = lidar.get_noisy_points(occupancy=occupancy_level)

        # project the pseuod-LIDAR onto the image plane and filter out depth points to make it sparse
        depth = lidar.project_into_image_and_draw_dist(depth, point_cloud, intr)
        
        depth_name = os.path.basename(path)

        # Save image as >old-image-w/o-png>_occup-<occupancy_level>.png
        cv.imwrite(f"{depth_name.replace('.png', '')}_occup-{int(occupancy_level*100)}.png", depth)


if __name__ == "__main__":
    main()