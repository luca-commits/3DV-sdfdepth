# adapted from https://github.com/EPFL-VILAB/omnidata
import argparse
import glob
import os.path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

torch.set_num_threads(32)

parser = argparse.ArgumentParser(description="Visualize output for depth or surface normals")

parser.add_argument("--omnidata_path", dest="omnidata_path", help="path to omnidata model")
parser.set_defaults(omnidata_path="/home/yuzh/Projects/omnidata/omnidata_tools/torch/")

parser.add_argument("--pretrained_models", dest="pretrained_models", help="path to pretrained models")
parser.set_defaults(pretrained_models="/home/yuzh/Projects/omnidata/omnidata_tools/torch/pretrained_models/")

parser.add_argument("--task", dest="task", help="normal or depth")
parser.set_defaults(task="NONE")

parser.add_argument("--img_path", dest="img_path", help="path to rgb image")
parser.set_defaults(im_name="NONE")

parser.add_argument("--output_path", dest="output_path", help="path to where output image should be stored")
parser.set_defaults(store_name="NONE")

args = parser.parse_args()

root_dir = args.pretrained_models
omnidata_path = args.omnidata_path

sys.path.append(args.omnidata_path)
# print(sys.path)
from data.transforms import get_transform
from modules.midas.dpt_depth import DPTDepthModel
from modules.unet import UNet

trans_topil = transforms.ToPILImage()
os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps')
# map_location = torch.device('mps')

# Fits the depth of d1 to be consistent with d0
def fit_depth(d0, d1, overlap):
    c = d0.shape[0]
    
    d0_fit = d0[:, c-overlap:].reshape(-1)
    d1_fit = d1[:, :overlap].reshape(-1)

    d0_fit = d0_fit.unsqueeze(-1)
    d1_fit = torch.stack([d1_fit, torch.ones_like(d1_fit)], axis=-1)

    params = torch.linalg.lstsq(d1_fit, d0_fit)
    
    d1_adjusted = params[0][0, 0] * d1.squeeze() + params[0][1, 0]
    
    return d1_adjusted.unsqueeze(-1)

# Fits the depth of n1 to be consistent with n0
def fit_normal(n0, n1, overlap):
    c = n0.shape[0]
    
    # Colour channel comes first
    n0_fit = n0[:, c-overlap:].permute([2, 0, 1]).reshape(3, -1)
    n1_fit = n1[:, :overlap].permute([2, 0, 1]).reshape(3, -1)

    n0_mean = torch.mean(n0_fit, dim=1)
    n1_mean = torch.mean(n1_fit, dim=1)

    n0_fit = n0_fit - n0_mean.unsqueeze(-1)
    n1_fit = n1_fit - n1_mean.unsqueeze(-1)

    # Special least squares to obtain a rotation-reflection matrix
    S = n1_fit @ n0_fit.T

    U, Sigma, V = torch.svd(S, some=False, compute_uv=True)
    Lambda = torch.eye(3)
    Lambda[-1, -1] = torch.det(V @ U.T)
    
    R = torch.matmul(V, torch.matmul(Lambda, U.T))
    
    # Residual
    t = n0_mean - R @ n1_mean
    
    # print(torch.matmul(R, n1_fit).shape, t.unsqueeze(-1).shape)
    # print(R, torch.linalg.norm(n0_fit - n1_fit), torch.linalg.norm(n0_fit - torch.matmul(R, n1_fit)))
    
    return (R @ n1.unsqueeze(-1)).squeeze()

# get target task and model
if args.task == "normal":
    image_size = 384

    pretrained_weights_path = os.path.join(root_dir, "omnidata_dpt_normal_v2.ckpt")
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)

elif args.task == "depth":
    image_size = 384
    pretrained_weights_path = os.path.join(root_dir, "omnidata_dpt_depth_v2.ckpt")  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone="vitb_rn50_384")  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose(
    [
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
    ]
)


def save_outputs(img_path, output_file_name):
    save_path = os.path.join(args.output_path, output_file_name.replace("_rgb", f"_{args.task}") + ".png")
    # print(f"Reading input {img_path} ...")
    img = Image.open(img_path)

    # --- NEW -----------------------------
    # How much two crops should overlap to do least squares fitting
    overlap = 120

    # Size of a crop
    c = 384
    h, w, _ = np.shape(img)

    target_h = c
    target_w = w * target_h // h

    transform = transforms.Compose(
        [
            transforms.Resize((target_h, target_w), interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ]
    )

    img = transform(img).to(device)

    n_crops = int(np.ceil(target_w / (c - overlap)))

    # For a given index of a crop in an image, gets the left alignment of the crop in an image
    get_left_align = lambda i: np.minimum(np.maximum(0, i * (c - overlap)), target_w - c)

    # Segregate the image into multiple overlapping crops
    crops = []
    for i in range(n_crops):
        left_align = get_left_align(i)
        crops.append(img[:, :, left_align:left_align+c])

    crops_batched = torch.stack(crops)

    with torch.no_grad():
        output_raw = model(crops_batched).cpu()

    # Output dimension handling and normalization
    if args.task == 'normal':
        # Colour channel comes last
        output_raw = torch.permute(output_raw, [0, 2, 3, 1])

        # Normalize to unit vectors (Model output is in [0, 1], but has to be in [-1, -1])
        output_batched = (2. * output_raw - 1.) / torch.linalg.vector_norm(2. * output_raw - 1., dim=-1).unsqueeze(-1)
    else:
        output_batched = output_raw.unsqueeze(-1)
    
    # Do least squares fitting to match the different maps
    output_adjusted = [ output_batched[0] ]
    for i in range(1, n_crops):
        if args.task == 'normal':
            output_adjusted.append(fit_normal(output_adjusted[i-1], output_batched[i], overlap))
        else:
            output_adjusted.append(fit_depth(output_adjusted[i-1], output_batched[i], overlap))

    # Put the maps back into one big map
    if args.task == 'normal':
        output = torch.zeros((target_h, target_w, 3))
    else:
        output = torch.zeros((target_h, target_w, 1))
    for i in range(0, n_crops):
        left_align = get_left_align(i)
        output[:, left_align:left_align+c] = output_adjusted[i]

    # Linearly interpolate between the maps to smoothen
    output_smoothed = output.clone().detach()
    for i in range(1, n_crops):
        # Left alignment of the crop to the left
        left_align_l = get_left_align(i-1)

        # Left image of the current crop
        left_align_r = get_left_align(i)

        # The overlap can be bigger than the defined overlap if the right-most crop had to be pushed left to fit into the image
        true_overlap = left_align_l + c - left_align_r

        weights = torch.linspace(0., 1., true_overlap).unsqueeze(0).unsqueeze(-1)

        # Linear interpolation
        output_smoothed[:, left_align_r:left_align_l + c] = (1. - weights) * output_adjusted[i-1][:, c-true_overlap:] \
                                                          + weights * output_adjusted[i][:, :true_overlap]

    # Smoothing has messed up unit vectors, renormalize
    if args.task == 'normal':
        output_smoothed /= torch.linalg.vector_norm(output_smoothed, dim=-1).unsqueeze(-1)
    # Now we have merged image ------------------------------------

    if args.task == "depth": ## This will not work anymore --
        output_smoothed = output_smoothed.clamp(0, 1).squeeze()

        np.save(save_path.replace(".png", ".npy"), output_smoothed.detach().cpu().numpy())
        plt.imsave(save_path, output_smoothed.detach().cpu().squeeze(), cmap="viridis")
    else:
        output_smoothed = output_smoothed.permute([2, 0, 1])
        
        np.save(save_path.replace(".png", ".npy"), output_smoothed.detach().cpu().numpy())
        trans_topil(output_smoothed).save(save_path)

    # print(f"Writing output {save_path} ...")


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in tqdm(glob.glob(args.img_path + "/*_rgb.png")):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()
