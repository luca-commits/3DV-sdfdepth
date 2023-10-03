import os
import argparse
import wandb

parser = argparse.ArgumentParser(description='Auxiliary training script for W&B Sweeps')
parser.add_argument('--data')
parser.add_argument('--sweep', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if args.sweep is not None:
    sweep_configuration = {
        "name": args.data.split("/")[-2],
        "method": "grid",
        "metric": {"name": "Eval Images Metrics Dict (all images)/rgb_depth_tradeoff", "goal": "minimize"},
        "parameters": {
            "camera_optimizer": {"values": ["off", "SO3xR3"]},
            "depth_loss_mult": {"values": [1, 0.5, 0.1, 0.05, 0.01]}
        }
    }

    def train():
        wandb.init()

        os.system(f'ns-train depth-nerfacto-huge \
            --vis wandb \
            --experiment-name $(basename {args.data})_{wandb.config.depth_loss_mult}_{wandb.config.camera_optimizer} \
            --steps-per-save 5000 \
            --max-num-iterations 10000 \
            --steps-per-eval-image 5000 \
            --steps-per-eval-all-images 1000 \
            --pipeline.model.depth-loss-mult {wandb.config.depth_loss_mult} \
            --pipeline.model.predict-normals False \
            --pipeline.datamanager.camera-optimizer.mode {wandb.config.camera_optimizer} \
            --pipeline.datamanager.images-on-gpu True \
            nerfstudio-data \
            --data {args.data} \
            --depth-unit-scale-factor 0.00390625')


    sweeps = wandb.Api().project("nerfstudio-project").sweeps()

    # only run sweep if there is no previous sweep of this scene
    if len([sweep for sweep in list(sweeps) if args.data.split("/")[-2] in sweep.name]) == 0:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="nerfstudio-project")
        wandb.agent(sweep_id, function=train, count=10)
    else:
        print("Sweep already completed. QUIT")
else:
    name = args.data.split("/")[-2]

    try:
        sweeps = list(wandb.Api().project("nerfstudio-project").sweeps())
        sweep = [ s for s in sweeps if name in s.config["name"] ][0]
        run = sweep.best_run()

        camera_optimizer = run.config["camera_optimizer"]
        depth_loss_mult = run.config["depth_loss_mult"]
    
    except:
        print("WARNING: Using default values for training")

        camera_optimizer = "off"
        depth_loss_mult = 0.1

    os.system(f'ns-train depth-nerfacto-huge \
        --vis wandb \
        --output-dir outputs-post-sweep \
        --experiment-name {name} \
        --steps-per-save 5000 \
        --max-num-iterations 30000 \
        --steps-per-eval-image 5000 \
        --steps-per-eval-all-images 1000 \
        --pipeline.model.depth-loss-mult {depth_loss_mult} \
        --pipeline.model.predict-normals False \
        --pipeline.datamanager.camera-optimizer.mode {camera_optimizer} \
        nerfstudio-data \
        --data {args.data} \
        --depth-unit-scale-factor 0.00390625')