import torch.nn as nn
import torch.nn.functional as F
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
