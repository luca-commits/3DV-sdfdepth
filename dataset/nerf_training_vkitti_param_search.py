import os
import argparse
import wandb

parser = argparse.ArgumentParser(description='Auxiliary training script for W&B Sweeps')
parser.add_argument('--data')
parser.add_argument('--sparseness', type=float, default=0)
parser.add_argument('--sweep', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if args.sweep is not None:
    sweep_configuration = {
        "name": args.data.split("/")[-1],
        "method": "grid",
        "metric": {"name": "Eval Images Metrics Dict (all images)/rgb_depth_tradeoff", "goal": "minimize"},
        "parameters": {
            "camera_optimizer": {"values": ["off", "SO3xR3"]},
            "depth_loss_mult": {"values": [1, 0.5, 0.1, 0.05, 0.01]}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="nerfstudio-project")

    def train():
        wandb.init()

        os.system(f'ns-train depth-nerfacto-huge \
            --output-dir /cluster/project/infk/courses/252-0579-00L/group26/vkitti/nerf_train_outputs \
            --vis wandb \
            --experiment-name $(basename {args.data})_sparseness_{args.sparseness}_{wandb.config.depth_loss_mult}_{wandb.config.camera_optimizer}\
            --steps-per-save 5000 \
            --max-num-iterations 30000 \
            --steps-per-eval-image 1000 \
            --steps-per-eval-all-images 1000 \
            --pipeline.model.depth-loss-mult {wandb.config.depth_loss_mult} \
            --pipeline.model.predict-normals False \
            --pipeline.datamanager.camera-optimizer.mode {wandb.config.camera_optimizer} \
            --pipeline.datamanager.images-on-gpu True \
            nerfstudio-data \
            --data {args.data} \
            --depth-unit-scale-factor 0.01')

    wandb.agent(sweep_id, function=train, count=10)
else:
    name = args.data.split("/")[-1]

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

    os.system(f'ns-train depth-nerfacto \
        --output-dir /cluster/project/infk/courses/252-0579-00L/group26/vkitti/nerf_train_outputs \
        --vis wandb \
        --output-dir nyu-outputs-post-sweep \
        --experiment-name {name}_sparseness_{args.sparseness} \
        --steps-per-save 5000 \
        --max-num-iterations 30000 \
        --steps-per-eval-image 5000 \
        --steps-per-eval-all-images 1000 \
        --pipeline.model.depth-loss-mult {depth_loss_mult} \
        --pipeline.model.predict-normals False \
        --pipeline.datamanager.camera-optimizer.mode {camera_optimizer} \
        nerfstudio-data \
        --data {args.data} \
        --depth-unit-scale-factor 0.01')