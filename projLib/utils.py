import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from tqdm import tqdm
import copy


def validate(model, valloader, train_args):

    device = train_args["device"]
    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction="sum")
    running_loss = 0.0

    all_predictions = torch.zeros(len(valloader.dataset))
    all_targets = torch.zeros(len(valloader.dataset))

    for i, (input, target, _) in enumerate(valloader, 0):
        
        #print(target.shape, input.shape, labels)
        if i == 0:
            start = i*input.shape[0]
        else:
            start = end
        end = start + input.shape[0]

        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(input)
            loss = criterion(outputs, target)
            running_loss += loss

            all_predictions[start:end] = outputs.to("cpu").squeeze()
            all_targets[start:end] = target.to("cpu").squeeze()
            

def train(model, trainloader, valloader, train_args):
    
    device = train_args["device"]
    model.to(device)

    optimizer = Adam(model.parameters(), **train_args["optimizer_args"])

    if train_args["scheduler"] == "LinearLR":
        scheduler = LinearLR(optimizer=optimizer,total_iters=50,verbose = False, start_factor= 0.5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience = 5)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = Adam(model.parameters(), **train_args["optimizer_args"])
    

    best_mse = 10e8

    result_dict = {}
    print(train_args["verbose"])    
    for epoch in tqdm(range(train_args["epochs"])):
        
        model.train()
        running_loss = 0.0
        all_predictions = torch.zeros(len(trainloader.dataset))
        all_targets = torch.zeros(len(trainloader.dataset))
        for i, (input, target, _) in enumerate(trainloader):

            input, target = input.float().to(device), target.float().to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input)
            loss = criterion(outputs, target)
            # loss = loss.float()
            loss.backward()
            optimizer.step()
            
            start = i*train_args["batch_size"]
            end = start + input.shape[0]
            print(f"rhs size: {outputs.detach().to('cpu').squeeze().size()}")
            all_predictions[start:end] = outputs.detach().to("cpu").squeeze() #
            all_targets[start:end] = target.to("cpu").squeeze()

            running_loss += loss.item()

        
        train_mse = running_loss/len(trainloader.dataset)
        val_mse, val_spearman, val_pearsoncorr = validate(model, valloader,train_args)

       
        
        if val_mse <= best_mse :
            best_mse = val_mse
            best_mse_model = copy.deepcopy(model)
            best_epoch = epoch
            if train_args["verbose"]:
                print(f"update in epoch {epoch}, {best_mse}", flush=True)
        
        if train_args["scheduler"] == "LinearLR":
            scheduler.step()

        
  

    result_dict["mse"] = best_mse
    result_dict["best_epoch"] = best_epoch
    result_dict["model"] = copy.deepcopy(best_mse_model)
    print("Best model", best_epoch, best_mse)

    return result_dict

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
