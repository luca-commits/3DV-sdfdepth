import os
import torch
from typing import Tuple


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
            print(scene_name)
            scene_date = "_".join(scene_name.split("_")[:3])
            if (scene_name in  ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync"]):
                #print(scene_date)
                camera_folders = list(os.scandir(os.path.join(folder, "proj_depth", "groundtruth")))
                for camera_folder in camera_folders:
                    print(camera_folder)
                    camera_name = os.path.basename(camera_folder)
                    for image in list(os.scandir(camera_folder)):
                        #print(image)
                        image_name = os.path.basename(image)
                        self.target_paths.append(image.path)
                        self.image_paths.append(os.path.join(img_dir, scene_date, scene_name, camera_name, "data", image_name))
        if self.keep_in_memory:
            self.images = []
            self.targets = []
            for image_path in self.image_paths:
                self.images.append(Image.open(image_path))
            for target_path in self.target_paths:
                self.targets.append(Image.open(target_path))

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
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        target = target / 1000.
        mask = target > 0
        return image.float(), target.float(), mask