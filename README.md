# Monocular Depth Estimation with Virtual-View Supervision

This code repository contains source code developed for the purposes of the project that was an integral part of the 2023 3D Vision course at ETH.

## Repository structure:

* `monocular_depth_estimation` - contains code for training and validating monocular depth prediction models
    * `eigen_<split>_files.txt` - text files listing the different splits of the dataset used by Eigen et al.
    * `datasets.py` - defines the pytorch dataset used for working with KITTI data
    * `loss.py` - defines the loss function for training our monocular depth prediction model
    * `models.py` - defines the model used (U-Net with a resnet18 backbone)
    * `utils.py` - utilities for training the model
    * `train.py` - model training script
* `depth completion` - contains code for getting completed depth maps from sparse ground truth KITTI depth maps
* `dataset` - contains utilities for manipulating the KITTI data
    * `extract_monocular_cues.py` - code for getting surface normal maps using the omnidata model
    * `generate_nerfstudio_dataset.py` - code for generating a dataset of images and corresponding camera intrinsics and extrinsics according to the nerfstudio format
    * `process_data.py` - driver script for generating scenes on which MonoSDF models will be trained
    * `process_nerfstudio_to_sdfstudio.py` - code for adding monocular cues and converting the scenes in nerfstudio format to a format compatible with SDF Studio
    * `submit_preprocessing.sh` - submission script for running jobs on Euler
    * `train_syncs.txt` - list of KITTI scenes which will be represented using MonoSDF models
* `training/sdfstudio` - contains scripts for training MonoSDF models and rendering novel views using the trained models
    * `render.py` - code for rendering novel views
    * `submit_training.sh` - script for invoking the training of a MonoSDF model for a particular scene
    * `render_views.sh` - script for invoking the rendering of a scene for which a MonoSDF model was trained

*** 

## Running the code

The steps outlined for running the code all assume execution on the ETH Euler cluster.

### 0. Download the KITTI dataset

Download the raw KITTI data using the download script at https://www.cvlibs.net/datasets/kitti/raw_data.php.

Download the KITTI depth data from https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

### 1. Install SDF Studio
Since Euler doesn't allow a straightforward installation of some of the dependencies required for SDF Studio, the installation process is more involved than described in the SDF Studio documentation.

To install SDF Studio and the necessary dependencies, run the following sequence of commands:

```
git clone https://github.com/autonomousvision/sdfstudio.git
cd sdfstudio
# open pyproject.toml with your favourite editor and change the line with open3d to "open3d==0.9.0"
env2lmod
module load gcc/8.2.0 cuda/11.6.2 open3d/0.9.0 python/3.8.5 eth_proxy
python -m venv --system-site-packages venv
source ./venv/bin/activate
pip install --ignore-installed torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade pip setuptools
export TCNN_CUDA_ARCHITECTURES="61, 70, 75, 80, 86" # CUDA architectures o/w not visible from login node
pip install --force-reinstall --upgrade git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
pip install python-snappy cryptography # some missing packages
```
The execution of the previous code block takes around an hour.

### 1. Install requirements

Install requirements listed in the `requirements.txt` file.

### 2. Install Omnidata

To be able to train the MonoSDF model, for each image in the KITTI dataset, we need corresponding surface normal maps. Since the surface normal maps themselves aren't part of the KITTI datasets, generating normal maps relies on the omnidata model.

Clone https://github.com/EPFL-VILAB/omnidata.git and follow the instructions in the repository to install Omnidata and download the pretrained models: https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#pretrained-models.

### 3. Depth Completion

The MonoSDF model used requires complete depth maps, but the ground truth depth data present in the KITTI dataset is sparse. Therefore, we need to complete the ground truth depth maps. For depth completion, we use the SemAttNet model available here: https://github.com/danishnazir/SemAttNet.

#### Steps to generate the completed depths for the KITTI dataset:

- Have the ground truth KITTI rgb and depth data downloaded.

- Download the KITTI semantic segmentation data from https://drive.google.com/file/d/1Yq-vcIuu9USrpKYc9J6Svu19QiEZYf92/view?usp=sharing and unzip into the `depth_completion/semantic_maps` directory.

- Download the SemAttNet trained model from https://drive.google.com/file/d/1plg4zGCLYndP0xtkh_gjG1RZ4YzPeiDN/view?usp=sharing and place it in the root of this folder.

- Call the `depth_completion/compile_continuous_depths.py` script and provide it with the following arguments:

    `--img-dir` - Path to directory with KITTI RGB images

    `--target-dir` - Path to directory with KITTI depth maps

- The comleted KITTI depth maps will be in the `depth_completion/cont_depth` directory

### 4. Generating the scene dataset

Define which KITTI scenes you want to train scene representation models by listing them in the `train_syncs.txt` file. To run the generation code, in the `submit_preprocessing.sh` script, change the following constants:

 - `OUTPUT_DIR`: directory where the generated scene datasets will be stored
 - `KITTI_RGB`: path to KITTI RGB images
 - `IMAGE_DIR`: directory which will be used to store intermediate dataset files
 - `COMPLETED_DEPTH_DIR`: directory containing completed depths (output of previous step)
 - `OMNIDATA_PATH`: path to the downloaded Omnidata repository
 - `PRETRAINED_OMNIDATA_MODELS`: path to directory containing pretrained Omnidata models
 - `VENV_PATH`: path to your virtual environment

After editing the constants, start the `submit_preprocessing.sh` script.

The scenes will be split into sections of 100 frames and each 100 frame section will be used to train a separate scene representation model. 

### 5. Training scene representation models

Once you have generated the scene dataset following the instructions oulined above, you are ready to start training the scene representation model (MonoSDF).

To train the scene representation model, edit the `training/sdfstudio/submit_training.sh` file and change the following constants to the desired values:

- `SCENE_NAME`: name of the scene (see example in the script)
- `VENV_PATH`: path to your virtual environment
- `SCENE_DATASETS_DIR`: directory where the generated scene datasets are be stored

The output of this phase of the pipeline are the trained scene representation models, which will be present in the `training/sdfstudio/output` directory.

### 6. Rendering novel views

Once you have trained the scene representation model, to render novel views for a particular scene, you are ready to render the novel views of that scene.

To render novel views, edit the `training/sdfstudio/render_views.sh` file and change the following constants to the desired values:

- `SCENE_NAME`: name of the scene (see example in the script)
- `MONOSDF_NAME`: name of the MonoSDF model (see example in the script)
- `TIMESTAMP`: timestamp of the MonoSDF model
- `ANGLE`: angle by which the novel views will be rotated around the Z axis
- `VENV_PATH`: path to your virtual environment
- `SCENE_DATASETS_DIR`: directory where the generated scene datasets are be stored

(hint: the `NERF_NAME` and `TIMESTAMP` can be found in the path to your saved model in the outputs folder, an example is given by the default values in the `render_views.sh` script)

The rendered scenes will be present in the `training/sdfstudio/renders` directory.

### 7. Training and evaluating the monocular depth prediction network

To train the monocular depth prediction model using the ground truth and novel views, run the `monocular_depth_estimation/train.py` script and provide it with the following arguments:

    - data-path: Path to the ground truth KITTI data
    - aug-dir: Path to the directory containing novel views 

The training script uses Weights and Biases to track experiments and log validation and testing results. If you want to have access to the outputs, make sure you are logged in to Weights and Biases before running the training script.

***

## External libraries

The majority of libraries used for the development of this project are listed in the `requirements.txt` file.

Repositories whose code is adapted in this repository are:
- SDF Studio, version 2023.03.12, https://github.com/autonomousvision/sdfstudio
- Omnidata, version Aug 2022, https://github.com/EPFL-VILAB/omnidata
- SemAttNet, version Oct 2022, https://github.com/danishnazir/SemAttNet
- AdaBins, version 1.0, https://github.com/shariqfarooq123/AdaBins
- UNet/FCN Pytorch, version 2022.08.21, https://github.com/usuyama/pytorch-unet

## Sample data

A sample of the data used is present in the `sample` directory. It includes the KITTI raw image data and the corresponding depth information.

## Authors

Niall Siegenheim, Luca Wolfart, Lovro Rabuzin, Mert Ertugrul

D-INFK, ETH Zurich