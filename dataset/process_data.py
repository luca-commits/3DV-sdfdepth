import os
import math

target_num_frames_per_clip = 100

with open('train_syncs.txt') as file:
    scenes = [line.rstrip() for line in file]
    
# scenes = ['2011_09_26_drive_0001_sync']

for scene in scenes:
    print(scene)

    os.makedirs(f"/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/{scene}", exist_ok=True)

    os.system(f"python generate_nerfstudio_dataset.py \
                --rgb-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images \
                --depth-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/depth/data_depth_annotated/train \
                --scene {scene} \
                --save-path /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/{scene}")