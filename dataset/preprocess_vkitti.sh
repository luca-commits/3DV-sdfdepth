#!/bin/bash

#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -A ls_polle
#SBATCH --job-name=vkitti_preprocessing
#SBATCH --output=vkitti_preprocessing_%j.out

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 ffmpeg/5.0 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

python process_vkitti.py
