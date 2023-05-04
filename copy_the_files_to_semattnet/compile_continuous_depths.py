import SemAttNetLoader
import os
import numpy as np
from model import A_CSPN_plus_plus
from model import three_branch_bb
import torch
from dataloaders.kitti_loader import  input_options, KittiDepth
from dataloaders.kitti_loader import rgb_read, depth_read, semantic_read
import argparse
from dataloaders import transforms
import tqdm as tqdm
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


#default=''
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')


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

print("Completed.")



print("=> creating model and optimizer ... ", end='')


model = A_CSPN_plus_plus(args).to(device)

model.load_state_dict(checkpoint['model'], strict=False)

data_loader = SemAttNetLoader(img_dir="~/Documents/3Dvision/3dv_sdfdepth/data/rgb_images",
                              target_dir="~/Documents/3Dvision/3dv_sdfdepth/data/data_depth_annotated",
                              semantic_dir="./semantic_maps",
                              batch_size=32
                              )

for i, (items, path) in tqdm(enumerate(data_loader), total=len(data_loader)):
    rgb_conf, semantic_conf, d_conf, rgb_depth, semantic_depth, d_depth, coarse_depth, pred = model(items)
    vis_utils.save_depth_as_uint16png_upload(pred, path)

