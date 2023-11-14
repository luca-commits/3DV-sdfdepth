import glob
import os.path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import mediapy as media

#CHANGE THESE FOR YOUR NEEDS
base_path = "/home/casimir/ETH/kitti/renders_reconstructed_post_sweep/2011_09_26_drive_0001_sync_0/pos_angle"
output_path = "/home/casimir/ETH/kitti/visuals"


os.makedirs(output_path, exist_ok=True)

for f in tqdm(glob.glob(base_path + "/*depth.png")):

    img_np = media.read_image(f)
    img = media.to_rgb(img_np, cmap="viridis")
    media.write_image(f"{output_path}/{os.path.basename(f)}", img)


#0,14,28,42
    