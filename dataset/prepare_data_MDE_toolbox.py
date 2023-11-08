import os
from pathlib import Path
from tqdm import tqdm

renders_dir = "/home/casimir/ETH/kitti/disturb_renders"
rgb_dest = "/home/casimir/ETH/kitti/rgb_images"
depth_dest = "/home/casimir/ETH/kitti/depth_mde"

original_files_file_path = "/home/casimir/ETH/kitti/kitti_eigen_train.txt"
files_file_path = "/home/casimir/ETH/kitti/kitti_eigen_novel_train.txt"

files_file = open(files_file_path, "w+")

# Copy over all original paths to rgbs an depths
with open(original_files_file_path) as original_files_file:
    for line in original_files_file:
        files_file.write(line)

files_file.write("\n")

models = os.listdir(renders_dir)

for model in tqdm(models):
    model_path = f"{renders_dir}/{model}"
    full_path = f"{model_path}"
    drive_split = model
    drive = drive_split[:-2]
    split = drive_split[-1]
    date = drive[:10]

    # Copy RGBs
    rgb_dest_full = f"{rgb_dest}/additional/{model}/"
    Path(rgb_dest_full).mkdir(parents=True, exist_ok=True)
    os.popen(f'cp {full_path}/*_rgb.png {rgb_dest_full}')

    rgb_files = sorted(os.listdir(rgb_dest_full))

    # print(rgb_files)
    # breakpoint()

    # Copy depths
    depth_dest_full = f"{depth_dest}/additional/{model}/"
    Path(depth_dest_full).mkdir(parents=True, exist_ok=True)
    os.popen(f'cp {full_path}/*_depth.png {depth_dest_full}')

    depth_files = sorted(os.listdir(depth_dest_full))

    print(len(depth_files), len(rgb_files))

    # Write to file of rgbs and depth paths for training
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        files_file.write(f"additional/{model}/{rgb_file} additional/{model}/{depth_file} 0.0\n")

files_file.close()