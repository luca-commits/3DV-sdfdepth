import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD:utils.py
=======
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from tqdm import tqdm
import copy
from math import sqrt
import os
import wandb


from loss import MaskedMSELoss

>>>>>>> dee621b2c7769bca9111f340413e7ce31ad9b5f1:projLib/utils.py
import matplotlib.pyplot as plt

def visualise_sample(image, depth, mask = None) -> None:
        if mask is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(3)
        else:
            fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(image.permute(1, 2, 0).numpy())
        ax1.set_title("RGB Image")
        ax2.imshow(depth.permute(1, 2, 0).numpy(), cmap="turbo")
        ax2.set_title("Depth map")
        if mask is not None:
            ax3.imshow(mask.permute(1, 2, 0).numpy())
            ax3.set_title("Mask")
<<<<<<< HEAD:utils.py
=======

def visualise_prediction(image, gt_depth, predicted_depth, save_location) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.tight_layout(pad=3)
        ax1.imshow(image.permute(1, 2, 0).numpy())
        ax1.set_title("RGB Image")
        ax2.imshow(gt_depth.permute(1, 2, 0).numpy(), cmap="turbo")
        ax2.set_title("Depth map")
        ax3.imshow(predicted_depth.permute(1, 2, 0).detach().numpy())
        ax3.set_title("Predicted depth")
        fig.savefig(save_location)


def validate(model, valloader, train_args):

    device = train_args["device"]
    model.to(device, non_blocking=True)
    model.eval()

    criterion = MaskedMSELoss()
    running_loss = 0.0

    for i, (input, target, mask) in enumerate(valloader, 0):
        
        if i == 0:
            start = i*input.shape[0]
        else:
            start = end
        end = start + input.shape[0]

        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type=str(device)):
                outputs = model(input)
                loss = criterion(outputs, target, mask)
                running_loss += loss

    return sqrt(running_loss/len(valloader))
            

def train(model, trainloader, valloader, train_args):
    
    device = train_args["device"]
    model.to(device, non_blocking=True)

    optimizer = Adam(model.parameters(), **train_args["optimizer_args"])

    if train_args["scheduler"] == "LinearLR":
        scheduler = LinearLR(optimizer=optimizer,total_iters=train_args["epochs"],verbose = False, start_factor= 0.5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience = 5)
    criterion = MaskedMSELoss()
    optimizer = Adam(model.parameters(), **train_args["optimizer_args"])
    

    best_mse = 10e8

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
                loss = criterion(outputs, target, mask)
            loss.backward()
            optimizer.step()
            
            start = i*train_args["batch_size"]
            end = start + input.shape[0]

            running_loss += loss.item()
        
        train_mse = sqrt(running_loss/len(trainloader.dataset))
        val_mse = validate(model, valloader,train_args)
        print(f"Train rmse: {train_mse}, Validation rmse: {val_mse}")
        wandb.log({"train/mse": train_mse, "val/mse": val_mse})
        
        if val_mse <= best_mse :
            best_mse = val_mse
            best_mse_model = copy.deepcopy(model)
            best_epoch = epoch
            if train_args["verbose"]:
                print(f"update in epoch {epoch}, {best_mse}", flush=True)
        
        if (epoch + 1) % train_args["save_steps"] == 0:
            save_model(best_mse_model)
        
        if train_args["scheduler"] == "LinearLR":
            scheduler.step()

    result_dict["mse"] = best_mse
    result_dict["best_epoch"] = best_epoch
    result_dict["model"] = copy.deepcopy(best_mse_model)

    print("Best model", best_epoch, best_mse)

    return result_dict

def save_model(model, modeltag = "unet.pth"):
    pretrained_path = f"{os.getcwd()}/pretrained"
    if not os.path.exists(pretrained_path):
        os.makedirs(pretrained_path)
    model_path = f"{pretrained_path}/{modeltag}"
    torch.save(model.state_dict(), model_path)
>>>>>>> dee621b2c7769bca9111f340413e7ce31ad9b5f1:projLib/utils.py
