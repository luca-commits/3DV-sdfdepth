import os
from glob import glob
import os.path
import numpy as np
from PIL import Image
from tqdm import tqdm

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


def get_mean_std(novel_data_path, train_rgb_path, train_split_path):

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

    
    rgb_sum = np.zeros((1,3))
    count = 0

    for f in tqdm(all_paths):
        img = np.asarray(Image.open(f), np.uint8)
        img_flattened = np.reshape(img, (-1,3))
        rgb_sum = rgb_sum + np.sum(img_flattened, axis=0)
        count += img_flattened.shape[0]

    rgb_mean = rgb_sum / count

    rgb_std = np.zeros((1,3))
    

    for f in tqdm(all_paths):
        img = np.asarray(Image.open(f), np.uint8)
        img_flattened = np.reshape(img, (-1,3))

        diff = img_flattened - rgb_mean
        diff_sqrd = np.square(diff)
        diff_sqrd_sum = np.sum(diff_sqrd, axis=0)
        rgb_std += diff_sqrd_sum

    rgb_std = np.sqrt(rgb_std / (count - 1))


    return rgb_mean, rgb_std


# mean, std = get_mean_std_vec(novel_data_path="/home/casimir/ETH/kitti/transl_renders", 
#                          train_rgb_path="/home/casimir/ETH/kitti/rgb_images",
#                          train_split_path="/home/casimir/ETH/kitti/kitti_eigen_train.txt")

mean, std = get_mean_std(novel_data_path="/home/casimir/ETH/kitti/transl_renders", 
                         train_rgb_path="/home/casimir/ETH/kitti/rgb_images",
                         train_split_path="/home/casimir/ETH/kitti/kitti_eigen_train.txt")

print("Mean:", mean, "Std:", std)