import os
from glob import glob
import json
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


def filter_models(csv_path, save_to_txt):

    lpips_cutoff = 0.3
    abs_rel_cutoff = 0.05

    good_models = []

    df = pd.read_csv(csv_path)

    for ind in df.index:

        if df['Eval Images Metrics Dict (all images)/lpips'][ind] < lpips_cutoff and \
            df['Eval Images Metrics Dict (all images)/depth_abs_rel'][ind] < abs_rel_cutoff:

            good_models.append(df['scene'][ind])

    print("Models remaining after filtering:", len(good_models))

    if save_to_txt:
        with open('filtered_models.txt', 'w') as fp:
            for model in good_models:
                fp.write("%s\n" % model)

    return good_models


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def filter_images(base_path, DELETE_IMAGES=False, INSPECTION_MODE=False, \
                  PRESET_THRESHOLD=20, VERBOSE=True, FILTER_BASED_ON_RGB=True, \
                    FILTER_BASED_ON_DEPTH=True, DEPTH_THRESHOLD=0.5):


  
    scene_paths = sorted(glob(os.path.join(base_path, '*')))

    for scene_path in tqdm(scene_paths):

        img_path = scene_path
        label_path = scene_path

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            raise Exception("Path doesn't exist")
        
        blur_metrics = {}

        if FILTER_BASED_ON_RGB:
            total_rgb_filtered_images = 0
            # for f in tqdm(sorted(glob(img_path + "/*rgb.png"))):
            for f in sorted(glob(img_path + "/*rgb.png")):
                img = np.asarray(Image.open(f), np.uint8)

                # breakpoint()

                # c = plt.imshow(img)
                # plt.show()
                
                blur = variance_of_laplacian(img)

                blur_metrics[f] = blur

            # print(blur_metrics)

            blur_metrics_np = np.array(list(blur_metrics.values()))

            blur_mean = blur_metrics_np.mean()
            blur_stdev = blur_metrics_np.std()
            
            # print(blur_mean)
            # print(blur_stdev)

            Z_score_threshold = -1.5
            # print(Z_score_threshold * blur_stdev + blur_mean)

            if INSPECTION_MODE:
                fig, axs = plt.subplots(1, 1)
                axs.hist(blur_metrics_np, bins=30)
                plt.show()

                manual_threshold = float(input("Please input threshold based on histogram: "))

            else:
                manual_threshold = PRESET_THRESHOLD
                
            for image in blur_metrics:
                blur = blur_metrics[image]
                # print(Z_score)

                if blur < manual_threshold:
                        total_rgb_filtered_images += 1

                        if DELETE_IMAGES:
                            os.remove(image)
                            os.remove(image.replace('rgb', 'depth').replace('.png', '.png'))

                        if VERBOSE:
                            print(blur)
                            print(image)

                        # c = plt.imshow(img)
                        # plt.show()

            print("RGB filter found: ", total_rgb_filtered_images)

        if FILTER_BASED_ON_DEPTH:
            total_depth_filtered_images = 0
            # for f in tqdm(sorted(glob(label_path + "/*depth.png"))):
            for f in sorted(glob(label_path + "/*depth.png")):
                depth = np.asarray(Image.open(f)) / 255.0

                avg_depth = depth.mean()

                if avg_depth < DEPTH_THRESHOLD:
                    total_depth_filtered_images += 1

                    # c = plt.imshow(depth)
                    # plt.show()

                    if VERBOSE:
                        print(f)

                    if DELETE_IMAGES:
                        os.remove(f)
                        os.remove(f.replace('depth', 'rgb').replace('.png', '.png'))
            
                # print(depth.shape)
                
                # c = plt.imshow(depth)
                # plt.show()
                # breakpoint()

            print("Depth filter found: ", total_depth_filtered_images)


# filter_models(csv_path="/home/casimir/ETH/3dv_sdfdepth/training/nerfstudio/eval_metrics.csv", save_to_txt=True)
filter_images(base_path="/home/casimir/ETH/kitti/interp_renders", DELETE_IMAGES=True, INSPECTION_MODE=False, \
                  PRESET_THRESHOLD=500, VERBOSE=False, FILTER_BASED_ON_RGB=True, \
                    FILTER_BASED_ON_DEPTH=True, DEPTH_THRESHOLD=0.5)