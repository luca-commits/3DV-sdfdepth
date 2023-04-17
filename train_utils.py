import wandb
import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import copy
from loss import MaskedMSELoss

def validate(model, valloader, config):

    device = config["device"]
    model.to(device)
    model.eval()

    criterion = MaskedMSELoss()
    running_loss = 0.0

    for input, target, mask in valloader:

        input, target, mask = input.to(device), target.to(device), mask.to(device)
        with torch.no_grad():
            outputs = model(input)
            loss = criterion(outputs, target, mask)
            running_loss += loss
        
    total_loss = running_loss/len(valloader.dataset)
    return total_loss
            

def train(model, trainloader, valloader, config):
    
    device = config["device"]
    model.to(device)

    optimizer = AdamW(model.parameters(), lr = config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    criterion = MaskedMSELoss()
    

    best_mse = 10e8

    result_dict = {}
    for epoch in tqdm(range(config["max_epochs"])):
        model.train()
        running_loss = 0.0
        for input, target, mask in trainloader:

            input, target, mask = input.float().to(device), target.float().to(device), mask.bool().to(device)

            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, target, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_mse = running_loss/len(trainloader.dataset)
        val_mse = validate(model, valloader, config)

        wandb.log({"train_mse": train_mse, "val_mse": val_mse})
        
        if val_mse <= best_mse :
            best_mse = val_mse
            best_mse_model = copy.deepcopy(model)
            best_epoch = epoch

        scheduler.step()

    result_dict["mse"] = best_mse
    result_dict["best_epoch"] = best_epoch
    result_dict["model"] = copy.deepcopy(best_mse_model)

    return result_dict