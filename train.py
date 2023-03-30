import pandas as pd
import numpy as np
import os
from dataset import HistoneMarksDataset
import torch
from torch.utils.data import DataLoader
from models import unet
from utils import *
import argparse

def main(modeltag: str):

    # load data
    data_path = f"{os.getcwd()}/data"
    version = "unnormalised_200000_bin_200"
    
    
    val_dataset = MonoDepthDataset(img_dir=os.join(data_path, "rgb_images"), target_dir=os.join(data_path, "data_depth_annotated"))
    train_dataset = MonoDepthDataset()



    train_data_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=512, shuffle=True)

    model = unet

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
    parser = argparse.ArgumentParser(description='Train CNN')
    parser.add_argument('--modeltag', type=str, required=True)
    args = parser.parse_args()
    main(modeltag=args.modeltag)