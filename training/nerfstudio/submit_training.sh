#!/bin/bash

#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A ls_polle 
#SBATCH --job-name=nerfstudio_training
#SBATCH --output=nerfstudio_training_%A_%a.out
#SBATCH --array=0-90

datasets=(/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr_distance/*/)
datasets=$(echo $datasets | xargs -n1 | sort | xargs)

data_path=${datasets[$SLURM_ARRAY_TASK_ID]}

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

python train.py --data $data_path