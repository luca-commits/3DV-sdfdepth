from pathlib import Path
import subprocess
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

root_dir = "comparisons"

subfolders = sorted([ f.path for f in os.scandir(root_dir) if f.is_dir() ])

for folder in tqdm(subfolders):

    name = folder.split("/")[-1]

    img0 = np.array(cv2.imread(folder + "/after.png", cv2.IMREAD_COLOR), dtype=np.dtype('uint8'))
    img1 = np.array(cv2.imread(folder + "/before.png", cv2.IMREAD_COLOR), dtype=np.dtype('uint8'))

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    h, w = img0.shape[: 2]

    skip = 7 # increase this to make it quicker, e.g. 14, 19, 21, 28 should work; some numbers might crash below.

    while (w + h) % skip != 0:
        skip += 1
    assert (w + h) % skip == 0
    print("Using skip:", skip)


    aux_path = Path(folder + "/aux")
    aux_path.mkdir(exist_ok=True)
    # if you want the line to go in the opposite direction then you can do tqdm(list(enumerate(range(-w, h, skip)))[::-1]) I think.
    for idx, didx in tqdm(list(enumerate(range(-w, h, skip)))):
        mask_top = np.triu(np.ones([h, w]).astype(bool), -didx)

        masked_img0 = np.array(img0)
        masked_img0[~mask_top] = 0
        masked_img1 = np.array(img1)
        masked_img1[mask_top] = 0

        masked_img = masked_img0 + masked_img1


        plt.imshow(masked_img)
        plt.axis('off')
        
        plt.tight_layout(h_pad=0, w_pad=-2.5)
        plt.savefig(aux_path / ("%03d.png" % idx), bbox_inches='tight', dpi=300)
        plt.close()

    subprocess.call(["ffmpeg", "-i", f"{str(aux_path)}/%03d.png", f"{folder}/{name}.mp4", "-y"])
    shutil.rmtree(aux_path)