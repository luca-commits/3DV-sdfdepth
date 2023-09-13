#!/bin/bash

#SBATCH -c 8
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A s_stud_infk
#SBATCH --job-name=novel_views
#SBATCH --output=novel_views_%A_%a.out
#SBATCH --array=0

angle=10

datasets=(/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/*/)
datasets=$(basename $(echo $datasets | xargs -n1 | sort | xargs))

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

models=(./outputs/$dataset/depth-nerfacto/*/)
models=$(echo $models | xargs -n1 | sort -r | xargs)

# Select most recent run
config_path=${models[0]}config.yml

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 ffmpeg/5.0 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

ns-render angled \
--load-config $config_path \
--rendered-output-names rgb depth \
--pose-source train \
--output-format raw-separate \
--output-path ./renders/$(basename ${models[0]})/pos_angle \
--angle $angle

ns-render angled \
--load-config $config_path \
--rendered-output-names rgb depth \
--pose-source train \
--output-format raw-separate \
--output-path ./renders/$(basename ${models[0]})/neg_angle \
--angle -$angle