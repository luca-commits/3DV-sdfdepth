import os
from pathlib import Path
from tqdm import tqdm
import glob


def prepare_train(original, novel, renders_dir, rgb_dir, depth_dir, original_files_file_path, files_file_path):

    files_file = open(files_file_path, "w+")

    if original:
        # Copy over all original paths to rgbs an depths
        with open(original_files_file_path) as original_files_file:
            for line in original_files_file:
                paths = line.split(' ')
                rel_rgb = paths[0]
                rel_depth = paths[1]
                focal = paths[2]
                if rel_depth == "None":
                    new_line = rgb_dir + "/" + rel_rgb + " " + rel_depth + " " + focal
                else:
                    new_line = rgb_dir + "/" + rel_rgb + " " + depth_dir + "/" + rel_depth + " " + focal
                files_file.write(new_line)

        # files_file.write("\n")
        # files_file.close()

    if novel:
        models = os.listdir(renders_dir)

        with open(original_files_file_path) as original_files_file:

            for model in tqdm(models, "a"):
                model_path = f"{renders_dir}/{model}"
                full_path = f"{model_path}"
                drive_split = model
                drive = drive_split[:-2]
                split = drive_split[-1]
                date = drive[:10]

                # rgb_dirs = [os.path.join(rgb_path, file) for file in sorted(os.listdir(rgb_path))]
                # depth_paths = [os.path.join(depth_paths, file) for file in sorted(os.listdir(depth_paths))]
                # rgb_paths = sorted(os.listdir(full_path +"/*_rgb.png"))
                # depth_paths = sorted(os.listdir(full_path +"/*_depth.png"))

                rgb_paths = sorted(glob.glob(full_path +"/*_rgb.png"))
                depth_paths = sorted(glob.glob(full_path +"/*depth.png"))

                for rgb_path, depth_path in zip(rgb_paths, depth_paths):

                    new_line = rgb_path + " " + depth_path + " " + str(0.0) + "\n"

                    files_file.write(new_line)

    files_file.close()


def prepare_test(original, novel, renders_dir, rgb_dir, depth_dir, original_files_file_path, files_file_path):

    files_file = open(files_file_path, "w+")

    if original:
        # Copy over all original paths to rgbs an depths
        with open(original_files_file_path) as original_files_file:
            for line in original_files_file:
                paths = line.split(' ')
                rel_rgb = paths[0]
                rel_depth = paths[1]
                focal = paths[2]
                
                if rel_depth == "None":
                    new_line = rgb_dir + "/" + rel_rgb + " " + rel_depth + " " + focal
                else:
                    new_line = rgb_dir + "/" + rel_rgb + " " + depth_dir + "/" + rel_depth + " " + focal
                files_file.write(new_line)

        # files_file.write("\n")
        # files_file.close()


    if novel:
        models = os.listdir(renders_dir)

        with open(original_files_file_path) as original_files_file:

            for model in tqdm(models, "a"):
                model_path = f"{renders_dir}/{model}"
                full_path = f"{model_path}"
                drive_split = model
                drive = drive_split[:-2]
                split = drive_split[-1]
                date = drive[:10]

                # rgb_dirs = [os.path.join(rgb_path, file) for file in sorted(os.listdir(rgb_path))]
                # depth_paths = [os.path.join(depth_paths, file) for file in sorted(os.listdir(depth_paths))]
                # rgb_paths = sorted(os.listdir(full_path +"/*_rgb.png"))
                # depth_paths = sorted(os.listdir(full_path +"/*_depth.png"))

                rgb_paths = sorted(glob.glob(full_path +"/*_rgb.png"))
                depth_paths = sorted(glob.glob(full_path +"/*depth.png"))

                for rgb_path, depth_path in zip(rgb_paths, depth_paths):

                    new_line = rgb_path + " " + depth_path + " " + str(0.0) + "\n"

                    files_file.write(new_line)

    files_file.close()


rgb_dir = "/home/casimir/ETH/kitti/rgb_images"
depth_dir = "/home/casimir/ETH/kitti/depth_mde"
original_files_file_path = "/home/casimir/ETH/kitti/kitti_eigen_train.txt"
files_file_path = "/home/casimir/ETH/kitti/kitti_eigen_novel_train.txt"
train_renders_dir = "/home/casimir/ETH/kitti/train_poses"

prepare_train(original=True, 
                novel=False, 
                renders_dir=train_renders_dir, 
                rgb_dir=rgb_dir, 
                depth_dir=depth_dir, 
                original_files_file_path=original_files_file_path, 
                files_file_path=files_file_path)


rgb_dir = "/home/casimir/ETH/kitti/rgb_images"
depth_dir = "/home/casimir/ETH/kitti/depth_mde"
original_files_file_path = "/home/casimir/ETH/kitti/kitti_eigen_test.txt"
files_file_path = "/home/casimir/ETH/kitti/kitti_eigen_novel_test.txt"
test_renders_dir = "/home/casimir/ETH/kitti/nerf_test_images"

prepare_test(original=True, 
                novel=False, 
                renders_dir=train_renders_dir, 
                rgb_dir=rgb_dir, 
                depth_dir=depth_dir, 
                original_files_file_path=original_files_file_path, 
                files_file_path=files_file_path)
