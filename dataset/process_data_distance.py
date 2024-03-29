import os
import math

with open('train_syncs.txt') as file:
    scenes = [line.rstrip() for line in file]

 
# scenes = ['2011_09_26_drive_0001_sync']

MAX_SCENE_SIZE = 50.0 #meters

for scene in scenes:
    date = scene[:10]
    drive = scene[-9:-5]
    
    print(date, drive)
    
    os.system(f"python generate_nerfstudio_dataset_distance.py \
                --rgb-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images \
                --depth-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/depth/data_depth_annotated/train \
                --scene {scene} \
                --save-path /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/{scene} \
                --max-scene-size {MAX_SCENE_SIZE}")

    # os.system(f"python generate_nerfstudio_dataset_distance.py \
    #             --rgb-base-path /home/casimir/ETH/kitti/rgb_images \
    #             --depth-base-path /home/casimir/ETH/kitti/depth/data_depth_annotated/train \
    #             --scene {scene} \
    #             --save-path /home/casimir/ETH/kitti/test/{scene} \
    #             --max-scene-size {MAX_SCENE_SIZE}")