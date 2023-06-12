import os
import math
import argparse

target_num_frames_per_clip = 100

def main(args):
    with open('train_syncs.txt') as file:
        scenes = [line.rstrip() for line in file]

    for scene in scenes:
        date = scene[:10]
        drive = scene[-9:-5]
        
        print(date, drive)
        
        if os.path.exists(f"{args.output_dir}/{scene}"):
            continue
        
        os.system(f"mkdir -p {args.image_dir}/{date}/")
        
        # Copy intrinsics files
        os.system(f"cp {args.kitti_rgb}/{date}/*.txt \
                    {args.image_dir}/{date}/")
        
        # Copy images
        os.system(f"cp -r {args.kitti_rgb}/{date}/{scene} \
                    {args.image_dir}/{date}/")
        
        # Get number of frames in scene
        num_frames = len(os.listdir(f'{args.image_dir}/{date}/{scene}/image_02/data/'))
        
        num_clips = math.ceil(num_frames / target_num_frames_per_clip)
        num_frames_per_clip = num_frames // num_clips
        
        for clip_idx in range(num_clips):
            # Generate Nerfstudio dataset
            start_frame = clip_idx * num_frames_per_clip
            end_frame = (clip_idx + 1) * num_frames_per_clip
            
            print(num_frames, start_frame, end_frame)
            
            os.system(f"python generate_nerfstudio_dataset.py \
                        --basedir {args.image_dir}/ \
                        --date {date} --drive {drive} --start_frame {start_frame} --end_frame {end_frame}")
            
            # Convert Nerfstudio to SDFStudio dataset, generating normals using Omnidata and copying the upsampled depths from the right location
            os.system(f"python process_nerfstudio_to_sdfstudio.py --data-type colmap --scene-type unbound \
                        --data {args.image_dir}/{date}/{scene}/ \
                        --depth-data {args.completed_depth_dir}/{date}/{scene}/ \
                        --output-dir {args.output_dir}/{scene}_{clip_idx}/ \
                        --omnidata-path {args.omnidata_path}/ \
                        --pretrained-models {args.pretrained_omnidata_models}/ \
                        --mono-prior")
        
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_paper', help='Output directory')
    parser.add_argument('--kitti-rgb', type=str, default='/cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images', help='KITTI RGB images directory')
    parser.add_argument('--image-dir', type=str, default='/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images', help='Temporary image directory')
    parser.add_argument('--completed-depth-dir', type=str, default='/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/completed_depth', help='Completed depth directory')
    parser.add_argument('--omnidata-path', type=str, default='/cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/torch/', help='Path to Omnidata repository')
    parser.add_argument('--pretrained-omnidata-models', type=str, default='/cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/torch/pretrained_models/', help='Path to pretrained Omnidata models')
    args = parser.parse_args()
    main(args)