import os

with open('train_syncs.txt') as file:
    scenes = [line.rstrip() for line in file]

for scene in scenes:
    date = scene[:10]
    drive = scene[-9:-5]
    
    print(date, drive)
    
    # if os.path.exists(f"/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_poster/{scene}"):
    #     continue
    
    # os.system(f"mkdir -p /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/")
    
    # Copy intrinsics files
    # os.system(f"cp /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images/{date}/*.txt \
    #             /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/")
    
    # Copy images
    # os.system(f"cp -r /cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images/{date}/{scene} \
    #             /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/")
    
    # Generate Nerfstudio dataset
    # os.system(f"python generate_nerfstudio_dataset.py \
    #             --basedir /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/ \
    #             --date {date} --drive {drive}")
    
    # Convert Nerfstudio to SDFStudio dataset, generating depths and normals using Omnidata 
    os.system(f"python process_nerfstudio_to_sdfstudio.py --data-type colmap --scene-type indoor \
                --data /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/{date}/{scene}/ \
                --depth-data /cluster/project/infk/courses/252-0579-00L/group26/depth_completition/stupid_models/SemAttNet/cont_depth/{date}/{scene}/ \
                --output-dir /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_poster/{scene}/ \
                --omnidata-path /cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/torch/ \
                --pretrained-models /cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/torch/pretrained_models/ \
                --mono-prior")
    
    print()