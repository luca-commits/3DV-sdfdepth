from SemAttNetLoader import SemAttDataset
import os
import numpy as np
from model import A_CSPN_plus_plus
from model import three_branch_bb
import torch
from dataloaders.kitti_loader import  input_options, KittiDepth
from dataloaders.kitti_loader import rgb_read, depth_read, semantic_read
import argparse
from dataloaders.transforms import *
from tqdm import tqdm
import vis_utils

parser = argparse.ArgumentParser(description='Sparse-to-Dense')

parser.add_argument('--data-folder',
                    default="",
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default="",
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('--network-model', default='', type=str)
parser.add_argument('--img-dir', default='', type=str, help='path to the rgb images')
parser.add_argument('--target-dir', default='', type=str, help='path to the target (depth) images')
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

args = parser.parse_args()

checkpoint = None
is_eval = False

model_path = os.path.join(os.getcwd(), "model_best_backup.pth.tar")

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))


checkpoint = torch.load(model_path, map_location=device)
print(f"current epoch: {checkpoint['epoch'] + 1}")

print("=> creating model and optimizer ... ", end='')

model = A_CSPN_plus_plus(args).to(device)

model.load_state_dict(checkpoint['model'], strict=False)

model.eval()

data_loader = torch.utils.data.DataLoader(SemAttDataset(args=args,
                              img_dir=args.img_dir,
                              target_dir=args.target_dir,
                              semantic_dir="./semantic_maps",
                              cont_depth_dir="./cont_depth"
                              ), batch_size=1, shuffle=False, num_workers=8, pin_memory=False)

with torch.no_grad():
    for i, (items, path) in tqdm(enumerate(data_loader), total=len(data_loader)):
        items = {item: items[item].to(device) for item in items}
        rgb_conf, semantic_conf, d_conf, rgb_depth, semantic_depth, d_depth, coarse_depth, pred = model(items)
        for i in range(len(path)):
            vis_utils.save_depth_as_uint16png(pred[i], path[i])
