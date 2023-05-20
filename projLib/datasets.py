import os
import torch
from typing import Tuple
from tqdm import tqdm
import numpy as np

from random import sample
from random import seed

seed(42)

from PIL import Image

class MonoDepthDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_dir: str,
                 target_dir: str = None,
                 aug_dir: str = None,
                 aug_len: int = -1,
                 transform: torch.nn.Module = None,
                 target_transform: torch.nn.Module = None,
                 keep_in_memory: bool = False,
                 image_list = None) -> None:
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform
        self.keep_in_memory = keep_in_memory
        self.aug_dir = aug_dir
        self.aug_len = aug_len
        self.no_aug_images = 0
        self.image_paths = []
        self.target_paths = []
        self.target_masks = []
        self.target_folders = list(os.scandir(target_dir))
        for partition in self.target_folders:
            data_folders = list(os.scandir(os.path.join(target_dir, partition)))
            for folder in data_folders: # 
                scene_name = os.path.basename(folder)
                scene_date = "_".join(scene_name.split("_")[:3])
                # if (scene_name in  ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync"]): #####
                camera_folders = list(os.scandir(os.path.join(folder, "proj_depth", "groundtruth")))
                for camera_folder in camera_folders:
                    camera_name = os.path.basename(camera_folder)
                    for image in list(os.scandir(camera_folder)):
                        image_name = os.path.basename(image)
                        if image_list is not None:
                            eigen_path = os.path.join(scene_date, scene_name, camera_name, "data", image_name)
                            pre, ext = os.path.splitext(eigen_path)
                            eigen_path = pre + ".jpg"
                            if eigen_path not in image_list:
                                continue
                        self.target_paths.append(image.path)
                        self.image_paths.append(os.path.join(img_dir, scene_date, scene_name, camera_name, "data", image_name))
        if self.aug_dir is not None and self.aug_len != 0:
            self.aug_image_paths = []
            self.aug_target_paths = []
            scenes = list(os.scandir(self.aug_dir))
            for scene in scenes:
                scene_name = os.path.basename(scene)
                scene_timestamps = list(os.scandir(scene))
                for timestamp in scene_timestamps:
                    angles = list(os.scandir(timestamp))
                    for angle in angles:
                        modalities = list(os.scandir(angle))
                        for modality in modalities:
                            images = [f for f in list(os.scandir(modality)) if os.path.isfile(f)]
                            if os.path.basename(modality) == "rgb":
                                for image in images:
                                    self.no_aug_images += 1
                                    self.aug_image_paths.append(image.path)
                            elif os.path.basename(modality) == "depth":
                                for image in images:
                                    self.aug_target_paths.append(image.path)
            if self.aug_len > 0:
                sampled_augs = sample(list(zip(self.aug_image_paths, self.aug_target_paths)), self.aug_len)
                sampled_aug_image_paths = [x[0] for x in sampled_augs]
                sampled_aug_target_paths = [x[1] for x in sampled_augs]
                self.image_paths = self.image_paths + sampled_aug_image_paths
                self.target_paths = self.target_paths + sampled_aug_target_paths
            elif self.aug_len == -1:
                self.image_paths = self.image_paths + self.aug_image_paths
                self.target_paths = self.target_paths + self.aug_target_paths

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
        target = target / (80. * 256.)
        mask = target > 0
        return image.float(), target.float(), mask