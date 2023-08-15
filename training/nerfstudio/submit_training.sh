#!/bin/bash

#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A s_stud_infk 
#SBATCH --job-name=nerfstudio_training
#SBATCH --output=nerfstudio_training.out
#SBATCH --error=nerfstudio_training.err


SCENE_NAME=2011_09_26_drive_0001_sync_1

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

hostname -i
ns-train depth-nerfacto \
--vis wandb \
--output-dir ./output \
--steps-per-save 5000 \
--max-num-iterations 150000 \
--steps-per-eval-image 1000 \
--pipeline.model.depth-loss-mult 0.5 \
sdfstudio-data \
--data /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/${SCENE_NAME} \
--include-mono-prior True \
--depth-unit-scale-factor 1. \
# --data /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images/2011_09_26/2011_09_26_drive_0001_sync


