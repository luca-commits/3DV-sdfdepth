#!/bin/bash

#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:1
#SBATCH -A ls_polle
#SBATCH --job-name=novel_views
#SBATCH --output=novel_views.out

angle=0

module purge
module load gcc/8.2.0 cuda/11.8.0 python/3.9.9 ffmpeg/5.0 eth_proxy
source /cluster/project/infk/courses/252-0579-00L/group26/nerfstudio/venv/bin/activate

datasets=(/cluster/project/infk/courses/252-0579-00L/group26/sniall/kitti/datasets_cvpr_distance/*/)
datasets=$(basename $(echo $datasets | xargs -n1 | sort | xargs))

for dataset in "${datasets[@]}"; do
    # wd=(/cluster/project/infk/courses/252-0579-00L/group26/lrabuzin/nerfstudio_training/)
    # wd=(/cluster/project/infk/courses/252-0579-00L/group26/cfeldmann/3dv_sdfdepth/training/nerfstudio/)
    wd=(./)

    cd $wd

    models=(./outputs-post-sweep/$(basename $dataset)/depth-nerfacto/*/)
    models=$(echo $models | xargs -n1 | sort | xargs)

    # Select most recent run
    config_path=${models[-1]}config.yml

    if grep -q "$(basename $dataset)," "eval_metrics.csv"; then
        # ns-render interpolate \
        # --load-config $config_path \
        # --rendered-output-names rgb depth \
        # --pose-source train \
        # --output-format raw-separate \
        # --output-path /cluster/project/infk/courses/252-0579-00L/group26/sniall/3dv_sdfdepth/training/nerfstudio/renders_interpolate_post_sweep/$(basename $dataset)/ \
        # --interpolation-steps 1 \
        # --order-poses False

        ns-render angled \
        --load-config $config_path \
        --rendered-output-names rgb depth \
        --pose-source train \
        --output-format raw-separate \
        --output-path /cluster/project/infk/courses/252-0579-00L/group26/sniall/3dv_sdfdepth/training/nerfstudio/renders_reconstructed_post_sweep/$(basename $dataset)/pos_angle \
        --angle $angle

        # ns-render angled \
        # --load-config $config_path \
        # --rendered-output-names rgb depth \
        # --pose-source train \
        # --output-format raw-separate \
        # --output-path /cluster/project/infk/courses/252-0579-00L/group26/sniall/3dv_sdfdepth/training/nerfstudio/renders_angled_3_post_sweep/$(basename $dataset)/neg_angle \
        # --angle -$angle
    else
        echo "Model does not meet requirements for rendering"
    fi
done
