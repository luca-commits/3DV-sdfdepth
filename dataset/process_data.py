import os
import math

# with open('train_syncs.txt') as file:
#     scenes = [line.rstrip() for line in file]

target_num_frames_per_clip = 100
    
scenes = ['2011_09_26_drive_0001_sync']

for scene in scenes:
    date = scene[:10]
    drive = scene[-9:-5]
    
    print(date, drive)


    num_frames = len(os.listdir(f'/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/{scene}/image_02/data/'))
    
    num_clips = math.ceil(num_frames / target_num_frames_per_clip)
    num_frames_per_clip = num_frames // num_clips

    for clip_idx in range(num_clips):
        # Generate Nerfstudio dataset
        start_frame = clip_idx * num_frames_per_clip
        end_frame = (clip_idx + 1) * num_frames_per_clip
        
        print(num_frames, start_frame, end_frame)

        os.makedirs(f"/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/{scene}_{clip_idx}", exist_ok=True)

        os.system(f"python generate_nerfstudio_dataset.py \
                    --rgb-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images \
                    --depth-base-path /cluster/project/infk/courses/252-0579-00L/group26/kitti/depth/data_depth_annotated/train \
                    --scene {scene} \
                    --save-path /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/{scene}_{clip_idx} \
                    --first_index {start_frame} --last_index {end_frame}")