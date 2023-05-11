import os

import numpy as np
import json
import pykitti

import argparse

rotmat = np.transpose(np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]))

def get_frame_dict_single_cam(file_path, data,i, camera):
    if camera == 2:
        calib = data.calib.T_cam2_imu
    elif camera == 3:
        calib = data.calib.T_cam3_imu

    transform_matrix = rotmat.dot(data.oxts[i].T_w_imu.dot(np.linalg.inv(calib)))
    transform_matrix[:, 0:3, 1:3] *= -1

    return  {
            "file_path": file_path,
            "transform_matrix": [ list(transform_matrix[i]) for i in range(4)]
        }


def get_frame_dict_cam3(file_path, data,i, cam3_intrinsics):

    transform_matrix = rotmat.dot(data.oxts[i].T_w_imu.dot(np.linalg.inv(data.calib.T_cam3_imu)))

    return  cam3_intrinsics | {
            "file_path": file_path,
            "transform_matrix": [ list(transform_matrix[i]) for i in range(4)]
        }



def parse_transform(basedir, date, drive, camera, start_frame_idx=0, end_frame_idx=None, stride=None):

    data = pykitti.raw(basedir, date, drive)


    folder = f'{basedir}/{date}'
    cam = folder + '/calib_cam_to_cam.txt'

    with open(cam) as file:
        lines = [line.rstrip() for line in file]

    all_calibs = {line.split()[0][:-1]:line.split()[1:] for line in lines}

    d = {
        "fl_x": float(all_calibs[f'P_rect_0{camera}'][0]),
        "fl_y": float(all_calibs[f'P_rect_0{camera}'][5]),
        "cx": float(all_calibs[f'P_rect_0{camera}'][2]),
        "cy": float(all_calibs[f'P_rect_0{camera}'][6]),
        "w": 1242,
        "h": 375,
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0
    }

    save_loc = f'{basedir}/{date}/{date}_drive_{drive}_sync/'

    folder = f"image_0{camera}/data"

    full_folder_path = os.path.join(save_loc, folder)
    all_frames = sorted(os.listdir(full_folder_path))

    frame_dicts = []

    if end_frame_idx is None:
        end_frame_idx = len(all_frames)

    for i in range(start_frame_idx, end_frame_idx):
        frame_dicts.append(get_frame_dict_single_cam(os.path.join(folder, all_frames[i]), data,i, camera))

    d["frames"] = frame_dicts

    with open(os.path.join(save_loc,'transforms.json'),'w') as f:
        json.dump(d,f,indent=4)




def parse_transform_multicam(basedir, date, drive, start_frame_idx=0, end_frame_idx=None, stride=None):

    data = pykitti.raw(basedir, date, drive)

    folder = f'{basedir}/{date}'
    cam = folder + '/calib_cam_to_cam.txt'

    with open(cam) as file:
        lines = [line.rstrip() for line in file]


    all_calibs = {line.split()[0][:-1]:line.split()[1:] for line in lines}


    cam2_intrinsics = {        # cam 2 intrinsics - cam 3 provided per frame in frames dict
        "fl_x": float(all_calibs[f'P_rect_02'][0]),
        "fl_y": float(all_calibs[f'P_rect_02'][5]),
        "cx": float(all_calibs[f'P_rect_02'][2]),
        "cy": float(all_calibs[f'P_rect_02'][6])}
    

    cam3_intrinsics = { #cam 3 
        "fl_x": float(all_calibs[f'P_rect_03'][0]),
        "fl_y": float(all_calibs[f'P_rect_03'][5]),
        "cx": float(all_calibs[f'P_rect_03'][2]),
        "cy": float(all_calibs[f'P_rect_03'][6])}
    


    d = cam2_intrinsics
    d = d | {
        "w": 1242,
        "h": 375,
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0
    }

    save_loc = f'{basedir}/{date}/{date}_drive_{drive}_sync/'

    folder_cam2 = f"image_02/data"
    full_cam2_folder_path = os.path.join(save_loc, folder_cam2)
    all_frames_cam2 = sorted(os.listdir(full_cam2_folder_path))

    folder_cam3 = f"image_03/data"
    full_cam3_folder_path = os.path.join(save_loc, folder_cam3)
    all_frames_cam3 = sorted(os.listdir(full_cam3_folder_path))

    frame_dicts = []

    if end_frame_idx is None:
        end_frame_idx = len(all_frames_cam2)

    # cam 2 frames
    for i in range(start_frame_idx, end_frame_idx):
        frame_dicts.append(get_frame_dict_single_cam(os.path.join(folder_cam2, all_frames_cam2[i]), data,i, 2))

    # cam 3 frames -- add cam intrinsics per frame
    for i in range(start_frame_idx, end_frame_idx):
        frame_dicts.append(get_frame_dict_cam3(os.path.join(folder_cam3, all_frames_cam3[i]), data,i, cam3_intrinsics))


    d["frames"] = frame_dicts

    with open(os.path.join(save_loc,'transforms.json'),'w') as f:
        json.dump(d,f,indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Nerfstudio dataset')
    parser.add_argument('--basedir', type=str, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--drive', type=str, required=True)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    args = parser.parse_args()
    if args.camera == 0:
        parse_transform_multicam(args.basedir, args.date, args.drive, args.start_frame, args.end_frame, args.stride)
    else:
        parse_transform(args.basedir, args.date, args.drive, args.camera, args.start_frame, args.end_frame, args.stride)




