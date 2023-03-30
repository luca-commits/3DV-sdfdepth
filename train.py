import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from models import get_unet
from train_utils import train
import argparse
import wandb
from datasets import MonoDepthDataset

def main(config):    
    val_dataset = MonoDepthDataset(img_dir=os.join(config["data_dir"],config["rgb_image_dir"]),
                                   target_dir=os.join(config["data_dir"], config["depth_val_image_dir"]))
    train_dataset = MonoDepthDataset(img_dir=os.join(config["data_dir"], config["rgb_image_dir"]),
                                     target_dir=os.join(config["data_dir"], config["depth_train_image_dir"]))

    val_data_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    train_data_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    model = get_unet(config["encoder_type"])

    results = train(model, train_data_loader, val_data_loader, config)

    pretrained_path = os.path.join(config["data_dir"], "pretrained")
    if not os.path.exists(pretrained_path):
        os.makedirs(pretrained_path)

    model_path = os.path.join(pretrained_path, f"{config['experiment_description']}_{config['encoder_type']}.pth")
    torch.save(results["model"].state_dict(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a monocular depth estimation network')

    parser.add_argument('--experiment_description', default='')
    parser.add_argument('--encoder_type', default='resnet34')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--data_dir', default='/cluster/scratch/lrabuzin/data')
    parser.add_argument('--rgb_image_dir', default='KITTI/raw_data')
    parser.add_argument('--depth_train_image_dir', default='KITTI/train')
    parser.add_argument('--depth_val_image_dir', default='KITTI/val')

    args = parser.parse_args()

    wandb.init(project="3DV_sdfdepth", config=vars(args))
    wandb.config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = wandb.config

    main(config)