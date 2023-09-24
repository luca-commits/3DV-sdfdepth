import os
import argparse
import wandb

parser = argparse.ArgumentParser(description='Auxiliary training script for W&B Sweeps')
parser.add_argument('--data')
args = parser.parse_args()

sweep_configuration = {
    "name": args.data.split("/")[-2],
    "method": "grid",
    "metric": {"name": "Eval Images Metrics Dict (all images)/rgb_depth_tradeoff", "goal": "minimize"},
    "parameters": {
        "camera_optimizer": {"values": ["off", "SO3xR3"]},
        "depth_loss_mult": {"values": [1, 0.5, 0.1, 0.05, 0.01]}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="nerfstudio-project")

def main():
    wandb.init()

    os.system(f'ns-train depth-nerfacto-huge \
    --vis wandb \
    --experiment-name $(basename {args.data})_{wandb.config.depth_loss_mult}_{wandb.config.camera_optimizer} \
    --steps-per-save 5000 \
    --max-num-iterations 10000 \
    --steps-per-eval-image 1000 \
    --steps-per-eval-all-images 1000 \
    --pipeline.model.depth-loss-mult {wandb.config.depth_loss_mult} \
    --pipeline.model.predict-normals False \
    --pipeline.datamanager.camera-optimizer.mode {wandb.config.camera_optimizer} \
    --pipeline.datamanager.images-on-gpu True \
    nerfstudio-data \
    --data {args.data} \
    --depth-unit-scale-factor 0.00390625')

wandb.agent(sweep_id, function=main, count=10)