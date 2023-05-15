#!/bin/bash

#SBATCH -c 8
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g
#SBATCH -A s_stud_infk 
#SBATCH --job-name=sdfstudio_training
#SBATCH --output=sdfstudio_training.out
#SBATCH --error=sdfstudio_training.err


SCENE_NAME=2011_09_26_drive_0001_sync

module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 intel-tbb/2020.3 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sdfstudio/venv/bin/activate

ns-train monosdf \
--output-dir ./output \
--trainer.steps-per-save 5000 \
--trainer.max-num-iterations 200000 \
--trainer.steps-per-eval-image 1000 \
--pipeline.model.sdf-field.inside-outside True \
--pipeline.model.sdf-field.use-grid-feature True \
--pipeline.model.background-model grid \
--pipeline.datamanager.train-num-images-to-sample-from 1 \
--pipeline.datamanager.train-num-times-to-repeat-images 0 \
--vis wandb  \
--experiment-name poster-${SCENE_NAME}-$(date +%F_%T) \
sdfstudio-data \
--data /cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_poster/${SCENE_NAME} \
--auto-orient True \
--include-mono-prior True