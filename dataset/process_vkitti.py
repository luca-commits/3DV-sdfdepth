import os
import math

target_num_frames_per_clip = 100
    
scenes = ['0001', '0002', '0006', '0018', '0020']
worlds = ['clone'] # 'fog', 'morning', 'overcast', 'rain', 'sunset'


for scene in scenes:
    for world in worlds:
        # os.makedirs(f"/cluster/project/infk/courses/252-0579-00L/group26/vkitti/nerfstudio_datasets/{scene}_{world}", exist_ok=True)

        os.system(f"python generate_nerfstudio_dataset_vkitti.py \
                    --rgb-base-path /cluster/project/infk/courses/252-0579-00L/group26/vkitti/vkitti_1.3.1_rgb \
                    --depth-base-path /cluster/project/infk/courses/252-0579-00L/group26/vkitti/vkitti_1.3.1_depthgt \
                    --extrinsic-base-path /cluster/project/infk/courses/252-0579-00L/group26/vkitti/vkitti_1.3.1_extrinsicsgt \
                    --scene {scene} \
                    --world {world} \
                    --save-path /cluster/project/infk/courses/252-0579-00L/group26/vkitti/nerfstudio_datasets/{world}/{scene} \
                    --max-scene-size 50")