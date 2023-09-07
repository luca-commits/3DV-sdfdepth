import numpy as np
import os
import argparse
import json
# import matplotlib.pyplot as plt

def quat2rotm(q):
    """Convert quaternion into rotation matrix """
    q /= np.sqrt(np.sum(q**2))
    x, y, z, s = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r1 = np.stack([1-2*(y**2+z**2), 2*(x*y-s*z), 2*(x*z+s*y)], axis=1)
    r2 = np.stack([2*(x*y+s*z), 1-2*(x**2+z**2), 2*(y*z-s*x)], axis=1)
    r3 = np.stack([2*(x*z-s*y), 2*(y*z+s*x), 1-2*(x**2+y**2)], axis=1)
    return np.stack([r1, r2, r3], axis=1)

def pose_vec2mat(pvec, use_filler=True):
    """Convert quaternion vector represention to SE3 group"""
    t, q = pvec[np.newaxis, 0:3], pvec[np.newaxis, 3:7]
    R = quat2rotm(q)
    t = np.expand_dims(t, axis=-1)
    P = np.concatenate([R, t], axis=2)
    if use_filler:
        filler = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 1, 4])
        P = np.concatenate([P, filler], axis=1)

    w2c = P[0]

    c2w = np.linalg.inv(w2c)
    
    # c2w[0:3, 1:3] *= -1
    # c2w = c2w[np.array([1, 0, 2, 3]), :]
    # c2w[2, :] *= -1

    rotmat = np.transpose(np.array([[0,  1, 0, 0],
                                            [-1, 0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]]))

    transform_matrix = rotmat.dot(w2c)
    transform_matrix[0:3, 1:3] *= -1

    #stackexchaneg advice
    # rotmat = np.array([[1, 0, 0, 0],
    #                     [0, -1, 0, 0],
    #                     [0,  0, -1, 0],
    #                     [0,  0, 0, 1]])
    # transform_matrix = np.transpose(rotmat.dot(mat))
    # # transform_matrix[0:3, 1:3] *= -1

    return c2w.tolist()


def orbslam_to_nerfstudio_matrix(row):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """

    # Extract the values from Q
    q0 = float(row[4])
    q1 = float(row[5])
    q2 = float(row[6])
    q3 = float(row[7])
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    ext_matrix =[[r00, r01, r02, float(row[1])],
              [r10, r11, r12, float(row[2])],
              [r20, r21, r22, float(row[3])],
              [0, 0, 0, 1]]
    
    ext_matrix = np.array(ext_matrix)

    rotmat = np.transpose(np.array([[-1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]]))
    ext_matrix = rotmat.dot(ext_matrix)
    ext_matrix[0:3, 1:3] *= -1
                            
    return ext_matrix.tolist()



def build_nyu_nerfstudio_dict(save_loc, img_folder, pose_file):
    # RGB Intrinsic Parameters
    metadata_dict = {
        "camera_model": "OPENCV",
        "fl_x": 5.1885790117450188e+02,
        "fl_y": 5.1946961112127485e+02,
        "cx": 3.2558244941119034e+02,
        "cy": 2.5373616633400465e+02,
        "w": 640,
        "h": 480,
        "k1": 2.0796615318809061e-01,
        "k2": -5.8613825163911781e-01,
        "p1": 7.2231363135888329e-04,
        "p2": 1.0479627195765181e-03,
        "frames": []
    }

    full_folder_path = os.path.join(save_loc, img_folder)
    with open(pose_file) as f:
        pose_lines = f.readlines()
    all_frames = sorted(os.listdir(full_folder_path))

    # x_list = []
    # y_list = []
    # z_list = []

    for line in pose_lines:
        timestamp = line.split(" ")[0]
        if timestamp + "rgb.png" not in all_frames or timestamp + "depth.png" not in all_frames:
            continue
        mat = pose_vec2mat(np.array([float(x) for x in line.split(" ")[1:]]) )
        # x_list.append(mat[0][3])
        # y_list.append(mat[1][3])
        # z_list.append(mat[2][3])

        metadata_dict["frames"].append({
            "file_path": timestamp + "rgb.png",
            "transform_matrix": mat,    #orbslam_to_nerfstudio_matrix(line.split(" ")),
            "depth_file_path": timestamp + "depth.png"
        })

    # print(f"x mean: {np.mean(x_list)}, y mean: {np.mean(y_list)}, z mean: {np.mean(z_list)} ")
    # print(f"x max: {np.max(x_list)}, y max: {np.max(y_list)}, z max: {np.max(z_list)} ")
    # print(f"x min: {np.min(x_list)}, y min: {np.min(y_list)}, z min: {np.min(z_list)} ")    
    # print(f"x std: {np.std(x_list)}, y std: {np.std(y_list)}, z std: {np.std(z_list)} ")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x_list, y_list, z_list)
    # plt.show()

    return metadata_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Nerfstudio dataset for NYU scene')
    parser.add_argument('--basedir', type=str, required=True)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--pose_file', type=str, required=True)
    args = parser.parse_args()

    metadata_dict = build_nyu_nerfstudio_dict(args.basedir, args.img_folder, args.pose_file)
    with open(os.path.join(args.basedir, args.img_folder,'transforms.json'),'w') as f:
        json.dump(metadata_dict,f,indent=4)
