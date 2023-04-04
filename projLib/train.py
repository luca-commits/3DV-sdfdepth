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
import torchvision.transforms as transforms

def main():

    # load data
    data_path = f"{os.getcwd()}/../data"

    transform = transforms.Compose(
    [transforms.ToTensor(),])
    
    val_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "data_depth_annotated/val"), transform=transform, target_transform=transform) #
    train_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "data_depth_annotated/train"), transform=transform, target_transform=transform) #



    train_data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    model = ResNetUNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_args = {
        "epochs": 50, 
        "device": device,
        "scheduler": "ReduceLROnPlateau",
        "optimizer_args": { "lr": 0.0005},
        "verbose": True,
    }

    print(train_args)
    results = train(model, train_data_loader, val_data_loader, train_args)

    pretrained_path = f"{os.getcwd()}/pretrained"
    if not os.path.exists(pretrained_path):
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(pretrained_path)

    model_path = f"{pretrained_path}/{modeltag}"
    torch.save(results["model"].state_dict(), model_path)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train CNN')
    # parser.add_argument('--modeltag', type=str, required=True)
    # args = parser.parse_args()
  #  main(modeltag=args.modeltag)
    main()