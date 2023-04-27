import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from tqdm import tqdm
import copy
from math import sqrt
import os
import wandb
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

from loss import SILogLoss

import matplotlib.pyplot as plt

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x, device='cpu'):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return x * std + mean


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt_orig, pred_orig):
    gt = gt_orig.cpu()
    pred = pred_orig.cpu()
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    abs_rel = (np.abs(gt - pred) / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = torch.log(pred.float()) - torch.log(gt.float())
    silog = np.sqrt((err ** 2).mean() - (err).mean() ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def visualise_sample(image, depth, mask = None) -> None:
        if mask is not None:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        else:
            fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(image.permute(1, 2, 0).numpy())
        ax1.set_title("RGB Image")
        ax2.imshow(depth.permute(1, 2, 0).numpy(), cmap="turbo")
        ax2.set_title("Depth map")
        if mask is not None:
            ax3.imshow(mask.permute(1, 2, 0).numpy())
            ax3.set_title("Mask")

def visualise_prediction(image, gt_depth, predicted_depth, save_location) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.tight_layout(pad=1.2)
        ax1.imshow(image.permute(1, 2, 0).numpy())
        ax1.set_title("RGB Image")
        max_depth = max(gt_depth.max().item(), predicted_depth.max().item())
        min_depth = max(gt_depth.min().item(), predicted_depth.min().item())
        ax2.imshow(gt_depth.permute(1, 2, 0).numpy(), vmax = max_depth, vmin=min_depth)
        ax2.set_title("Depth map")
        im = ax3.imshow(predicted_depth.permute(1, 2, 0).detach().numpy(), vmax = max_depth, vmin=min_depth)
        ax3.set_title("Predicted depth")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax = cbar_ax)
        fig.savefig(save_location, dpi=600)


def validate(model, valloader, train_args):

    device = train_args["device"]
    model.to(device, non_blocking=True)
    model.eval()

    criterion = SILogLoss()
    running_loss = 0.0

    metrics = RunningAverageDict()

    for i, (input, target, mask) in enumerate(valloader, 0):

        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type=str(device)):
                outputs = model(input)
                loss = criterion(outputs, target, mask, False)
                running_loss += loss.item()
                metrics.update(compute_errors(target[mask], outputs[mask]))

    return {k: round(v.item(), 3) for k, v in metrics.get_value().items()} | {"val_loss": running_loss/len(valloader)}
            

def train(model, trainloader, valloader, train_args):
    
    device = train_args["device"]
    model.to(device, non_blocking=True)

    optimizer = Adam(model.parameters(), **train_args["optimizer_args"])

    if train_args["scheduler"] == "LinearLR":
        scheduler = LinearLR(optimizer=optimizer,total_iters=train_args["epochs"],verbose = False, start_factor= 0.5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience = 5)
    criterion = SILogLoss()
    optimizer = Adam(model.parameters(), **train_args["optimizer_args"])
    

    best_loss = 10e13

    result_dict = {}  
    for epoch in range(train_args["epochs"]):
        model.train()
        running_loss = 0.0
        for i, (input, target, mask) in tqdm(enumerate(trainloader), total=len(trainloader)):
            input, target = input.float().to(device), target.float().to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # forward + backward + optimize
            with torch.autocast(device_type=str(device)):
                outputs = model(input)
                # print("outputs")
                # print(outputs)
                loss = criterion(outputs, target, mask, False)
                # print(loss)
            wandb.log({"training batch loss": loss})
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), train_args["clip"])

            optimizer.step()

            running_loss += loss.item()
        
        train_loss = running_loss/len(trainloader)
        eval_metrics = validate(model, valloader,train_args)
        print(f"Train loss: {train_loss}, Validation loss: {eval_metrics['val_loss']}")
        wandb.log({"train/loss": train_loss, "val/loss": eval_metrics['val_loss']})
        wandb.log(eval_metrics)
        print(eval_metrics)
        
        if eval_metrics['val_loss'] <= best_loss or epoch==0:
            best_loss = eval_metrics['val_loss']
            best_loss_model = copy.deepcopy(model)
            best_epoch = epoch
            if train_args["verbose"]:
                print(f"update in epoch {epoch}, {best_loss}", flush=True)
        
        if (epoch) % train_args["save_steps"] == 0:
            save_model(best_loss_model)
            pretrained_path = f"{os.getcwd()}/pretrained"
            best_loss_model.eval().to("cpu")
            with torch.no_grad():
                visualise_prediction(torch.stack((trainloader.dataset[0][0][0]*std[0]+mean[0],trainloader.dataset[0][0][1]*std[1]+mean[1],trainloader.dataset[0][0][2]*std[2]+mean[2])), trainloader.dataset[0][1], best_loss_model(trainloader.dataset[0][0].unsqueeze(0))[0], os.path.join(pretrained_path, "vis_train.png"))
                visualise_prediction(torch.stack((valloader.dataset[0][0][0]*std[0]+mean[0],valloader.dataset[0][0][1]*std[1]+mean[1],valloader.dataset[0][0][2]*std[2]+mean[2])), valloader.dataset[0][1], best_loss_model(valloader.dataset[0][0].unsqueeze(0))[0], os.path.join(pretrained_path, "vis_val.png"))
        
        if train_args["scheduler"] == "LinearLR":
            scheduler.step()

    result_dict["loss"] = best_loss
    result_dict["best_epoch"] = best_epoch
    result_dict["model"] = copy.deepcopy(best_loss_model)

    print("Best model", best_epoch, best_loss)

    return result_dict

def save_model(model, modeltag = "unet.pth"):
    pretrained_path = f"{os.getcwd()}/pretrained"
    if not os.path.exists(pretrained_path):
        os.makedirs(pretrained_path)
    model_path = f"{pretrained_path}/{modeltag}"
    torch.save(model.state_dict(), model_path)