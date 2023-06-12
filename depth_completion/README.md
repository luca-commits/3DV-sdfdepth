# Depth Completion Module

All code in this folder was adapted from https://github.com/danishnazir/SemAttNet.

## Steps to generate the completed depths for the KITTI dataset:

0. Have the ground truth KITTI rgb and depth data downloaded.

1. Download the KITTI semantic segmentation data from https://drive.google.com/file/d/1Yq-vcIuu9USrpKYc9J6Svu19QiEZYf92/view?usp=sharing and unzip into the `semantic_maps` directory.

2. Download the SemAttNet trained model from https://drive.google.com/file/d/1plg4zGCLYndP0xtkh_gjG1RZ4YzPeiDN/view?usp=sharing and place it in the root of this folder.

3. Call the `compile_continuous_depths.py` script and provide it with the following arguments:

    `--img-dir` - Path to directory with KITTI RGB images

    `--target-dir` - Path to directory with KITTI depth maps

4. The comleted KITTI depth maps will be in the `cont_depth` subdirectory