#!/bin/bash

#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A ls_polle 
#SBATCH --job-name=nerfstudio_training
#SBATCH --output=nerfstudio_training.out
#SBATCH --error=nerfstudio_training.err

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

array=(*/)
for dir in "${array[@]}"; do
    SCENE_NAME=$dir

    ns-train depth-nerfacto-huge \
    --vis wandb \
    --experiment-name ${SCENE_NAME} \
    --steps-per-save 5000 \
    --max-num-iterations 30000 \
    --steps-per-eval-image 1000 \
    --pipeline.model.depth-loss-mult 1.0 \
    --pipeline.model.predict-normals False \
    --pipeline.datamanager.camera-optimizer.mode off \
    nerfstudio-data \
    --data /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr/${SCENE_NAME} \
    --depth-unit-scale-factor 0.00390625 \
    echo "Finished training ${SCENE_NAME}"
done