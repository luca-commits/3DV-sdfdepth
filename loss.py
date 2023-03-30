import torch
import torch.nn as nn

def masked_mse_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    out = (output[mask]-target[mask])**2
    loss = out.mean()
    return loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, output, target, mask):
        return masked_mse_loss(output, target, mask)