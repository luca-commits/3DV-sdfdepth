from glob import glob
import yaml
import os
from nerfstudio.engine.trainer import TrainerConfig
from pathlib import Path

def average_results(models_dir, new_config_dir, method_name):

    models = [os.path.join(models_dir, file) for file in sorted(os.listdir(models_dir))]

    print("Number of models:", len(models))

    for model in models:
        run_dirs = glob(model + f"/{method_name}/*/")

        for run_dir in run_dirs:
            # latest_subdir = max(runs, key=os.path.getmtime)

            yaml_path = Path(run_dir) / Path("config.yml")

            # with open(json_path, 'r') as stream:
            #     data_loaded = yaml.safe_load(stream)

            config = yaml.load(yaml_path.read_text(), Loader=yaml.Loader)
            assert isinstance(config, TrainerConfig)


            config.output_dir = Path(models_dir) 
            config.pipeline.datamanager.dataparser.data = Path(new_config_dir) / Path(Path(model).parts[-1])

            print(config.output_dir)
            print(config.pipeline.datamanager.dataparser.data)
            
            yaml_path.write_text(yaml.dump(config), "utf8")


models_dir = "/home/casimir/ETH/kitti/tuned_scenes"
method_name = "depth-nerfacto"
transforms_dir = "/home/casimir/ETH/kitti/eigen_scenes"

average_results(models_dir, transforms_dir, method_name)