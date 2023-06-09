#!/bin/bash

#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A s_stud_infk 
#SBATCH --job-name=novel_views
#SBATCH --output=novel_views.out
#SBATCH --error=novel_views.err

SCENE_NAME=2011_09_26_drive_0015_sync
NERF_NAME=poster-2011_09_26_drive_0015_sync-2023-05-15_15\:59\:35
TIMESTAMP=2023-05-15_155956
ANGLE=15

module load gcc/8.2.0 ffmpeg/5.0 cuda/11.6.2 python/3.8.5 open3d/0.9.0 intel-tbb/2020.3 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sdfstudio/venv/bin/activate

python ../../render.py \
--load-config output/${NERF_NAME}/monosdf/${TIMESTAMP}/config.yml \
--metadata-path /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_paper/${SCENE_NAME}/meta_data.json \
--angle ${ANGLE} \
--output-path renders/${SCENE_NAME}/${TIMESTAMP}/${ANGLE}.png \
--downscale-factor 1.44