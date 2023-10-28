import os
from glob import glob
import os.path
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_mean_std(novel_data_path, train_rgb_path, train_split_path):

    novel_rgb_sum = np.zeros((1,3))

    #First get novel mean:
    scene_paths = sorted(glob(os.path.join(novel_data_path, '*')))

    # scene_paths = ['/home/casimir/ETH/kitti/transl_renders/2011_09_26_drive_0001_sync_0']

    novel_image_count  = 0

    for scene_path in tqdm(scene_paths):

        img_path = scene_path

        if not os.path.exists(img_path):
            raise Exception("Path doesn't exist")
        
        for f in sorted(glob(img_path + "/*rgb.png")):
            novel_image_count += 1

            img = np.asarray(Image.open(f), np.uint8)
            img_flattened = np.reshape(img, (-1,3))
            novel_rgb_sum = novel_rgb_sum + np.mean(img_flattened, axis=0)


    novel_rgb_mean = novel_rgb_sum / novel_image_count


    # Next get train_mean

    train_rgb_sum = np.zeros((1,3))

    with open(train_split_path) as fp:
        num_lines = len(fp.readlines())

    with open(train_split_path) as fp:

        line = fp.readline()
        for _ in tqdm(range(num_lines)):
            rgb_path = line.split(" ")[0]

            img = np.asarray(Image.open(f"{train_rgb_path}/{rgb_path}"), np.uint8)
            img_flattened = np.reshape(img, (-1,3))
            train_rgb_sum = train_rgb_sum + np.mean(img_flattened, axis=0) 

            line = fp.readline()
    

    train_rgb_mean = train_rgb_sum / num_lines

    total_mean = (novel_rgb_mean + train_rgb_mean) / 2

    #Now calculate std:

    total_std = np.zeros((1,3))

    for scene_path in tqdm(scene_paths):

        img_path = scene_path

        if not os.path.exists(img_path):
            raise Exception("Path doesn't exist")
        
        for f in sorted(glob(img_path + "/*rgb.png")):
            img = np.asarray(Image.open(f), np.uint8)
            img_flattened = np.reshape(img, (-1,3))

            diff = img_flattened - total_mean
            diff_sqrd = np.square(diff)
            diff_sqrd_sum = np.sum(diff_sqrd, axis=0)
            diff_sqrd_sum = diff_sqrd_sum / img_flattened.shape[0]
            total_std += diff_sqrd_sum

    with open(train_split_path) as fp:

        line = fp.readline()
        for _ in tqdm(range(num_lines)):
            rgb_path = line.split(" ")[0]

            img = np.asarray(Image.open(f"{train_rgb_path}/{rgb_path}"), np.uint8)
            img_flattened = np.reshape(img, (-1,3))

            diff = img_flattened - total_mean
            diff_sqrd = np.square(diff)
            diff_sqrd_sum = np.sum(diff_sqrd, axis=0)
            diff_sqrd_sum = diff_sqrd_sum / img_flattened.shape[0]
            total_std += diff_sqrd_sum

            line = fp.readline()


    total_std = np.sqrt(total_std / (num_lines + novel_image_count - 1))

    return total_mean, total_std


def get_mean_std_vec(novel_data_path, train_rgb_path, train_split_path):

    #First get novel mean:
    all_paths = []

    scene_paths = sorted(glob(os.path.join(novel_data_path, '*')))

    for scene_path in tqdm(scene_paths):

        img_path = scene_path

        if not os.path.exists(img_path):
            raise Exception("Path doesn't exist")
        
        for f in sorted(glob(img_path + "/*rgb.png")):
            all_paths.append(f)


    with open(train_split_path) as fp:
        num_lines = len(fp.readlines())

    with open(train_split_path) as fp:

        line = fp.readline()
        for _ in tqdm(range(num_lines)):
            rgb_path = line.split(" ")[0]
            all_paths.append(f"{train_rgb_path}/{rgb_path}")

    

    rgb_values = np.concatenate(
        [Image.open(img).getdata() for img in tqdm(all_paths)], 
        axis=0
    )

    mean_rgb = np.mean(rgb_values, axis=0)
    std_rgb = np.std(rgb_values, axis=0)

    return mean_rgb, std_rgb

mean, std = get_mean_std(novel_data_path="/home/casimir/ETH/kitti/transl_renders", 
                         train_rgb_path="/home/casimir/ETH/kitti/rgb_images",
                         train_split_path="/home/casimir/ETH/kitti/kitti_eigen_train.txt")

# mean, std = get_mean_std_vec(novel_data_path="/home/casimir/ETH/kitti/transl_renders", 
#                          train_rgb_path="/home/casimir/ETH/kitti/rgb_images",
#                          train_split_path="/home/casimir/ETH/kitti/kitti_eigen_train.txt")

print("Mean:", mean, "Std:", std)