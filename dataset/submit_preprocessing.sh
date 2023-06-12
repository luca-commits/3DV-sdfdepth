#!/bin/bash

#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A s_stud_infk
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.out

OUTPUT_DIR=/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_paper
KITTI_RGB=/cluster/project/infk/courses/252-0579-00L/group26/kitti/rgb_images
IMAGE_DIR=/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/images
COMPLETED_DEPTH_DIR=/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/completed_depth
OMNIDATA_PATH=/cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/torch/
PRETRAINED_OMNIDATA_MODELS=/cluster/project/infk/courses/252-0579-00L/group26/omnidata/omnidata_tools/pretrained_models/

module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 intel-tbb/2020.3 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sdfstudio/venv/bin/activate

python process_data.py --output-dir ${OUTPUT_DIR} \
--kitti-rgb ${KITTI_RGB} \
--image-dir ${IMAGE_DIR} \
--completed-depth-dir ${COMPLETED_DEPTH_DIR} \
--omnidata-path ${OMNIDATA_PATH} \
--pretrained-omnidata-models ${PRETRAINED_OMNIDATA_MODELS}
