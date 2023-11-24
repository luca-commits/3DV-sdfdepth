#!/bin/bash

#SBATCH -c 2
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH -A ls_polle
#SBATCH --job-name=render_masks_multi
#SBATCH --output=mask_log/angled_render_masks_%A_%a.out
#SBATCH --array=0-62

# renders_interpolate_post_sweep
cd /cluster/project/infk/courses/252-0579-00L/group26/nihars_tests/repo/3dv_sdfdepth/dataset

poses_path=/cluster/project/infk/courses/252-0579-00L/group26/sniall/3dv_sdfdepth/training/nerfstudio/renders_angled_3_post_sweep/
# dirs=$(ls -d $poses_path*/)

dirs=()
while IFS= read -r -d '' dir; do
    dirs+=("$dir")
done < <(find "$poses_path" -maxdepth 1 -type d -print0)

dir="${dirs[${SLURM_ARRAY_TASK_ID}]}/random.txt"
# dir="${dirs[4]}/random.txt"
dir="$(dirname "$dir")"

echo $dir
# echo $poses_path
# module purge
# module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 eth_proxy
# source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

python render_masks_multiproc.py ${poses_path} ${dir} 