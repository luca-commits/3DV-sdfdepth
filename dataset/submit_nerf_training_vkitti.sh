#!/bin/bash

#SBATCH -c 8
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH -A ls_polle 
#SBATCH --job-name=nerfstudio_training_vkitti
#SBATCH --output=vkitti_nerfstudio_training.out

BASE_DIR="/cluster/project/infk/courses/252-0579-00L/group26/vkitti/nerfstudio_datasets"
sparseness=0
world="clone"
subscenes=("0001_0" "0001_1")

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

# Loop through each scene and process
for subscene in "${subscenes[@]}"; do
    data_path="$BASE_DIR/$world/$subscene/cropped"
    python nerf_training_vkitti_param_search.py --data $data_path --sparseness $sparseness --sweep
done