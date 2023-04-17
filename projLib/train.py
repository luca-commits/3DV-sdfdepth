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
    [transforms.ToTensor(), transforms.Resize([384, 1248])])
    
    val_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "data_depth_annotated/val"), transform=transform, target_transform=transform) #
    train_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "data_depth_annotated/train"), transform=transform, target_transform=transform) #

    model = ResNetUNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_args = {
        "epochs": 1, 
        "device": device,
        "scheduler": "ReduceLROnPlateau",
        "optimizer_args": { "lr": 0.0005},
        "verbose": True,
        "batch_size": 2
    }

    train_data_loader = DataLoader(train_dataset, batch_size=train_args["batch_size"], shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=train_args["batch_size"], shuffle=True)

    print(train_args)
    results = train(model, train_data_loader, val_data_loader, train_args)

    pretrained_path = f"{os.getcwd()}/pretrained"
    if not os.path.exists(pretrained_path):
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(pretrained_path)

    modeltag = "unet.pth"

    model_path = f"{pretrained_path}/{modeltag}"
    torch.save(results["model"].state_dict(), model_path)

    results["model"].eval().to("cpu")
    with torch.no_grad():
        visualise_prediction(train_dataset[0][0], train_dataset[0][1], results["model"](train_dataset[0][0].unsqueeze(0))[0], os.path.join(pretrained_path, "vis.png"))



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train CNN')
    # parser.add_argument('--modeltag', type=str, required=True)
    # args = parser.parse_args()
  #  main(modeltag=args.modeltag)
    main()