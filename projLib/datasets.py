import os
import torch
from typing import Tuple
from tqdm import tqdm
import numpy as np

from PIL import Image

class MonoDepthDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_dir: str,
                 target_dir: str = None,
                 transform: torch.nn.Module = None,
                 target_transform: torch.nn.Module = None,
                 keep_in_memory: bool = False) -> None:
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform
        self.keep_in_memory = keep_in_memory
        self.image_paths = []
        self.target_paths = []
        self.target_masks = []
        self.target_folders = list(os.scandir(target_dir))
        for folder in self.target_folders: # 
            scene_name = os.path.basename(folder)
            scene_date = "_".join(scene_name.split("_")[:3])
            if (scene_name in  ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync"]):
                camera_folders = list(os.scandir(os.path.join(folder, "proj_depth", "groundtruth")))
                for camera_folder in camera_folders:
                    camera_name = os.path.basename(camera_folder)
                    for image in list(os.scandir(camera_folder)):
                        image_name = os.path.basename(image)
                        self.target_paths.append(image.path)
                        self.image_paths.append(os.path.join(img_dir, scene_date, scene_name, camera_name, "data", image_name))
        if self.keep_in_memory:
            self.images = []
            self.targets = []
            print("loading images")
            for image_path in tqdm(self.image_paths):
                self.images.append(self.transform(Image.open(image_path)))
            print("loading targets")
            for target_path in tqdm(self.target_paths):
                self.targets.append(self.target_transform(Image.open(target_path)))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        if self.keep_in_memory:
            image = self.images[idx]
            target = self.targets[idx]
        else:
            img_path = self.image_paths[idx]
            target_path = self.target_paths[idx]
            image = Image.open(img_path)
            target = Image.open(target_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            target = np.asarray(target, dtype=np.float32)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                target = self.target_transform(target)
        target = target / 256.
        mask = target > 0
        return image.float(), target.float(), mask