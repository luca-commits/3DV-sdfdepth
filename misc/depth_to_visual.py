import glob
import os.path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import mediapy as media

#CHANGE THESE FOR YOUR NEEDS


def waymo_eval_vis():
    base_path = "/home/casimir/ETH/kitti/waymo_visuals/depth/binsformer/waymo"
    output_path = "/home/casimir/ETH/kitti/waymo_visuals/binsformer/visual"

    for dir in os.listdir(base_path):

        os.makedirs(f"{output_path}/{dir}_viridis", exist_ok=True)

        rgb_dirs = [os.path.join(f"{base_path}/{dir}", file) for file in sorted(os.listdir(f"{base_path}/{dir}"))]

        for rgb_dir in tqdm(rgb_dirs):

            img_np = media.read_image(rgb_dir)

            # img_np = img_np / 256.0
            img = media.to_rgb(img_np, cmap="viridis", vmin=0.0, vmax=80.0)
            media.write_image(f"{output_path}/{dir}_viridis/{os.path.basename(rgb_dir)}", img)


def folder_vis(scale=1.0):
    base_path = "/home/casimir/ETH/Monocular-Depth-Estimation-Toolbox/visuals/kitti_depthformer_baseline"
    output_path = "/home/casimir/ETH/kitti/nerf_visuals/kitti_depthformer_baseline"

    os.makedirs(output_path, exist_ok=True)

    rgb_dirs = [os.path.join(base_path, file) for file in sorted(os.listdir(base_path))]

    for rgb_dir in tqdm(rgb_dirs):

        if "depth" in rgb_dir:

            img_np = np.asarray(media.read_image(rgb_dir), dtype=float)
            
            img_np /= scale

            # img_np = img_np / 256.0
            img = media.to_rgb(img_np, cmap="viridis", vmin=0.0, vmax=80.0)
            media.write_image(f"{output_path}/{os.path.basename(rgb_dir)}".replace(".png", "_depth.png"), img)

def kitti_renders_vis(angles):
    base_path = "/home/casimir/ETH/kitti/renders_reconstructed_post_sweep"
    output_path = "/home/casimir/ETH/kitti/nerf_visuals/renders_reconstructed_post_sweep"

    if len(angles) > 0:

        for dir in os.listdir(base_path):

            for angle in angles:

                os.makedirs(f"{output_path}/{dir}/{angle}", exist_ok=True)

                rgb_dirs = [os.path.join(f"{base_path}/{dir}/{angle}", file) for file in sorted(os.listdir(f"{base_path}/{dir}/{angle}"))]

                for rgb_dir in tqdm(rgb_dirs):

                    if "depth" in rgb_dir:

                        img_np = media.read_image(rgb_dir)

                        img_np = img_np / 256.0
                        img = media.to_rgb(img_np, cmap="viridis", vmin=0.0, vmax=80.0)
                        media.write_image(f"{output_path}/{dir}/{angle}/{os.path.basename(rgb_dir)}", img)

    else:
        for dir in os.listdir(base_path):

            os.makedirs(f"{output_path}/{dir}", exist_ok=True)

            rgb_dirs = [os.path.join(f"{base_path}/{dir}", file) for file in sorted(os.listdir(f"{base_path}/{dir}"))]

            for rgb_dir in tqdm(rgb_dirs):

                if "depth" in rgb_dir:

                    img_np = media.read_image(rgb_dir)

                    img_np = img_np / 256.0
                    img = media.to_rgb(img_np, cmap="viridis", vmin=0.0, vmax=80.0)
                    media.write_image(f"{output_path}/{dir}/{os.path.basename(rgb_dir)}", img)


waymo_eval_vis()


# folder_vis(scale=255)
# folder_vis(scale=1.0)

# angles=["pos_angle", "neg_angle"]
# angles=["pos_angle"]
# angles = []
# kitti_renders_vis(angles=angles)

