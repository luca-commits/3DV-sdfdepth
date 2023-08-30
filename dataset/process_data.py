import os
import math

target_num_frames_per_clip = 100

with open('train_syncs.txt') as file:
    scenes = [line.rstrip() for line in file]
    
scenes = ['2011_09_26_drive_0001_sync']

for scene in scenes:
    date = scene[:10]
    drive = scene[-9:-5]
    
    print(date, drive)
    
    # if os.path.exists(f"/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_paper/{scene}"):
    #     continue
    # 
    # os.system(f"mkdir -p /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/")
    # 
    # # Copy intrinsics files
    # os.system(f"cp /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images/{date}/*.txt \
    #             /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/")
    # 
    # # Copy images
    # os.system(f"cp -r /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images/{date}/{scene} \
    #             /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/")
    
    # Get number of frames in scene
    num_frames = len(os.listdir(f'/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/{scene}/image_02/data/'))
    
    num_clips = math.ceil(num_frames / target_num_frames_per_clip)
    num_frames_per_clip = num_frames // num_clips
    
    for clip_idx in range(num_clips):
        # Generate Nerfstudio dataset
        start_frame = clip_idx * num_frames_per_clip
        end_frame = (clip_idx + 1) * num_frames_per_clip
        
        print(num_frames, start_frame, end_frame)
        
        os.system(f"python generate_nerfstudio_dataset.py \
                    --basedir /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/ \
                    --date {date} --drive {drive} --start_frame {start_frame} --end_frame {end_frame}")
        
        # Convert Nerfstudio to SDFStudio dataset, generating normals using Omnidata and copying the upsampled depths from the right location
        os.system(f"python process_nerfstudio_to_sdfstudio.py --data-type colmap --scene-type unbound \
                    --data /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/{scene}/ \
                    --depth-data /cluster/project/infk/courses/252-0579-00L/group26/kitti/depth/data_depth_annotated/train/{scene}/proj_depth/groundtruth/ \
                    --output-dir /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/{scene}_{clip_idx}/ \
                    --omnidata-path /cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/torch/ \
                    --pretrained-models /cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/torch/pretrained_models/ \
                    --mono-prior")
    
    print()