#!/bin/bash

#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A s_stud_infk
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.out

module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 intel-tbb/2020.3 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/sdfstudio/venv/bin/activate

python process_data.py
