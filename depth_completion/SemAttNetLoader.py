import os
import torch
from typing import Tuple
from tqdm import tqdm

from PIL import Image
from dataloaders.transforms import *
from dataloaders.kitti_loader import rgb_read, depth_read, semantic_read
from torchvision import transforms
from dataloaders.kitti_loader import *

class SemAttDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 img_dir: str,
                 target_dir: str,
                 semantic_dir: str,
                 cont_depth_dir: str, ## directory where you will put the continuous depth
                 transform: torch.nn.Module = None,
                 target_transform: torch.nn.Module = None,
                 keep_in_memory: bool = False,
                 ) -> None:
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.semantics_dir = semantic_dir
        self.cont_depth_dir = cont_depth_dir
        self.transform = transform
        self.target_transform = target_transform
        self.keep_in_memory = keep_in_memory
        self.image_paths = []
        self.target_paths = []
        self.semantic_paths = []
        self.cont_depth_paths = []
        self.target_masks = []
        self.target_folders = list(os.scandir(target_dir))
        self.args = args
        for folder in self.target_folders: 
            scene_name = os.path.basename(folder)
            scene_date = "_".join(scene_name.split("_")[:3])
            camera_folders = list(os.scandir(os.path.join(folder, "proj_depth", "groundtruth")))
            for camera_folder in camera_folders:
                camera_name = os.path.basename(camera_folder)
                cont_depth_folder_path = os.path.join(cont_depth_dir, scene_date, scene_name, camera_name, "data")
                if (not os.path.exists(cont_depth_folder_path)):
                    os.makedirs(cont_depth_folder_path)
                for image in list(os.scandir(camera_folder)):
                    image_name = os.path.basename(image)
                    self.target_paths.append(image.path)
                    self.image_paths.append(os.path.join(img_dir, scene_date, scene_name, camera_name, "data", image_name))
                    self.semantic_paths.append(os.path.join(semantic_dir, scene_date, scene_name, camera_name, "data", image_name))
                    cont_depth_path = os.path.join(cont_depth_folder_path, image_name)
                    self.cont_depth_paths.append(cont_depth_path)
        if self.keep_in_memory:
            self.images = []
            self.targets = []
            self.semantics = []
            print("loading images")
            for image_path in tqdm(self.image_paths):
                self.images.append(self.transform(Image.open(image_path)))
            print("loading targets")
            for target_path in tqdm(self.target_paths):
                self.targets.append(self.target_transform(Image.open(target_path)))
            print("loading semantics")
            for semantic_path in tqdm(self.semantic_paths):
                self.semantics.append(Image.open(semantic_path))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        to_tensor = ToTensor()
        to_float_tensor = lambda x: to_tensor(x).float()

        img_path = self.image_paths[idx]
        target_path = self.target_paths[idx]
        semantic_path = self.target_paths[idx]
        rgb, semantic, sparse, target, position = val_transform(rgb_read(img_path), semantic_read(semantic_path), depth_read(target_path), None, None, self.args)
        sparse = sparse
        candidates = {"rgb": rgb, "semantic":semantic,  "d": sparse, "gt": target, 'position': position}    
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }
        
        return items, self.cont_depth_paths[idx]
