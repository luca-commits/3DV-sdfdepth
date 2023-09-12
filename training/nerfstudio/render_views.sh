#!/bin/bash

#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A s_stud_infk 
#SBATCH --job-name=novel_views
#SBATCH --output=novel_views.out
#SBATCH --error=novel_views.err

TIMESTAMP=2023-09-08_154019
ANGLE=180

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 ffmpeg/5.0 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

ns-render interpolate \
--load-config output/unnamed/depth-nerfacto/${TIMESTAMP}/config.yml \
--rendered-output-names rgb depth \
--pose-source train \
--output-format video \
# --angle 180
# --colormap-options.colormap-max 0.2