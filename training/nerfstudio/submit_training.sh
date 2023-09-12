#!/bin/bash

#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A s_stud_infk 
#SBATCH --job-name=nerfstudio_training
#SBATCH --output=nerfstudio_training_%A_%a.out
#SBATCH --array=40-49

datasets=(/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/*/)
datasets=$(echo $datasets | xargs -n1 | sort | xargs)

data_path=${datasets[$SLURM_ARRAY_TASK_ID]}

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

ns-train depth-nerfacto-huge \
--vis wandb \
--experiment-name $(basename $data_path) \
--steps-per-save 5000 \
--max-num-iterations 30000 \
--steps-per-eval-image 1000 \
--pipeline.model.depth-loss-mult 1.0 \
--pipeline.model.predict-normals False \
--pipeline.datamanager.camera-optimizer.mode off \
nerfstudio-data \
--data $data_path \
--depth-unit-scale-factor 0.00390625 \