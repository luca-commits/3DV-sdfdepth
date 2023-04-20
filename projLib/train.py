import pandas as pd
import numpy as np
import os
from datasets import MonoDepthDataset
import torch
from torch.utils.data import DataLoader
from models import ResNetUNet
from utils import *
import argparse
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import InterpolationMode
import wandb
import torchvision.transforms as transforms

def main():
    torch.autograd.detect_anomaly()
    # load data
    data_path = f"{os.getcwd()}/../kitti"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize([256, 768]), transforms.Normalize(mean, std, inplace=True)])
    target_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize([256, 768], InterpolationMode.NEAREST, antialias=False)])
    
    val_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "data_depth_annotated/val"), transform=transform, target_transform=target_transform) #
    train_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "data_depth_annotated/train"), transform=transform, target_transform=target_transform) #

    model = ResNetUNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    train_args = {
        "epochs": 15,
        "device": device,
        "scheduler": "LinearLR",
        "optimizer_args": { "lr": 0.0005},
        "verbose": True,
        "batch_size": 16,
        "save_steps": 3,
        "clip": 2.0
    }

    wandb.init(project="3DV_sdfdepth", config=train_args, anonymous="allow")
    wandb.watch(model, log='all')

    train_data_loader = DataLoader(train_dataset, batch_size=train_args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=train_args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    print(train_args)
    torch.backends.cudnn.benchmark = True
    results = train(model, train_data_loader, val_data_loader, train_args)

    save_model(results["model"])

    pretrained_path = f"{os.getcwd()}/pretrained"
    results["model"].eval().to("cpu")
    with torch.no_grad():
        visualise_prediction(train_dataset[0][0], train_dataset[0][1], results["model"](train_dataset[0][0].unsqueeze(0))[0], os.path.join(pretrained_path, "vis.png"))



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train CNN')
    # parser.add_argument('--modeltag', type=str, required=True)
    # args = parser.parse_args()
  #  main(modeltag=args.modeltag)
    main()