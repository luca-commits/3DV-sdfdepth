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

AUG_SIZE = -1

def main():
    # torch.autograd.detect_anomaly()
    # load data
    data_path = "/cluster/project/infk/courses/252-0579-00L/group26/kitti"
    aug_dir = "/cluster/project/infk/courses/252-0579-00L/group26/renders-all"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_files = load_eigen_train(os.path.join(os.getcwd(), "eigen_train_files.txt"))
    val_files = load_eigen_split(os.path.join(os.getcwd(), "eigen_val_files.txt"))
    test_files = load_eigen_split(os.path.join(os.getcwd(), "eigen_test_files.txt"))

    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize([256, 768]), transforms.Normalize(mean, std, inplace=True)])
    target_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize([256, 768], InterpolationMode.NEAREST, antialias=False)])
    
    val_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "depth/data_depth_annotated"), transform=transform, target_transform=target_transform, image_list=val_files) #
    train_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "depth/data_depth_annotated"), transform=transform, target_transform=target_transform, image_list=train_files, aug_dir=aug_dir, aug_len=AUG_SIZE) #
    test_dataset = MonoDepthDataset(img_dir=os.path.join(data_path, "rgb_images"), target_dir=os.path.join(data_path, "depth/data_depth_annotated"), transform=transform, target_transform=target_transform, image_list=test_files) #
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")
    print(f"test dataset size: {len(test_dataset)}")

    model = ResNetUNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    train_args = {
        "epochs": 20,
        "device": device,
        "scheduler": "LinearLR",
        "optimizer_args": { "lr": 0.0005},
        "verbose": True,
        "batch_size": 32,
        "save_steps": 3,
        "clip": 2.0,
        "aug_size": train_dataset.aug_len if AUG_SIZE == -1 else AUG_SIZE,
    }

    wandb.init(project="3DV_sdfdepth", config=train_args, anonymous="allow")
    wandb.watch(model, log='all')

    train_data_loader = DataLoader(train_dataset, batch_size=train_args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=train_args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=train_args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    print(train_args)
    torch.backends.cudnn.benchmark = True
    results = train(model, train_data_loader, val_data_loader, train_args)

    test_results = validate(model, test_data_loader, train_args)
    test_results = {f"test/{k}": v for k, v in test_results.items()}
    wandb.log(test_results)

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