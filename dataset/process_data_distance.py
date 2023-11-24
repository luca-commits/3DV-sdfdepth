import os
import math

sweeps_path = '/cluster/project/infk/courses/252-0579-00L/group26/nihars_tests/outputs-post-sweep'

all_dirs = [dire.split("sync")[0] + "sync" for dire in os.listdir(sweeps_path) if os.path.isdir(os.path.join(sweeps_path, dire))]

scenes = set(all_dirs)

# with open('train_syncs.txt') as file:
#     scenes = [line.rstrip() for line in file]

breakpoint()
 
#scenes = ['2011_09_26_drive_0001_sync']

USERNAME = 'nihars_tests'

MAX_SCENE_SIZE = 50.0 # meters

for scene in scenes:
    date = scene[:10]
    drive = scene[-9:-5]
    
    print(date, drive)
    
    os.system(f"python generate_nerfstudio_dataset_distance.py \
                --rgb-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images \
                --depth-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/depth/data_depth_annotated/train \
                --scene {scene} \
                --save-path /cluster/project/infk/courses/252-0579-00L/group26/{USERNAME}/kitti/datasets_cvpr/{scene} \
                --max-scene-size {MAX_SCENE_SIZE}")

    # os.system(f"python generate_nerfstudio_dataset_distance.py \
    #             --rgb-base-path /home/casimir/ETH/kitti/rgb_images \
    #             --depth-base-path /home/casimir/ETH/kitti/depth/data_depth_annotated/train \
    #             --scene {scene} \
    #             --save-path /home/casimir/ETH/kitti/test/{scene} \
    #             --max-scene-size {MAX_SCENE_SIZE}")