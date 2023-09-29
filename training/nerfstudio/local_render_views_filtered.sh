#!/bin/bash

angle=10

wd=(/home/casimir/ETH/3dv_sdfdepth/training/nerfstudio)

cd $wd

filter_list="filtered_models.txt"
suffix="_distance"
output_folder=/home/casimir/ETH/kitti/renders_filtered
mkdir -p output_folder


i=0
lower_bound=0
upper_bound=100

while IFS= read -r experiment_name; do

    scene_name=${experiment_name/%$suffix}

    models=(/home/casimir/ETH/3dv_sdfdepth/training/nerfstudio/outputs/${experiment_name}/depth-nerfacto/*/)
    models=$(echo $models | xargs -n1 | sort | xargs)

    config_path=${models[-1]}config.yml

    if [[ ("$i" -ge "$lower_bound") && ("$i" -lt "$upper_bound")]]
        then
            echo "Index $i: Rendering $scene_name"

            ns-render angled \
            --load-config $config_path \
            --rendered-output-names rgb depth \
            --pose-source train \
            --output-format raw-separate \
            --output-path $output_folder/$scene_name/pos_angle \
            --angle $angle

            ns-render angled \
            --load-config $config_path \
            --rendered-output-names rgb depth \
            --pose-source train \
            --output-format raw-separate \
            --output-path $output_folder/$scene_name/neg_angle \
            --angle -$angle
            echo "Finished training ${scene_name}"
    fi
    ((i++))
done < $filter_list


