#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never
from scipy.spatial.transform import Slerp, Rotation as R
from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

import os

CONSOLE = Console(width=120)

def get_rotmat(theta):
    theta = theta * np.pi / 180

    rotmat = np.transpose(np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]]))
    return rotmat

def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "images",
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                render_image.append(output_image)
            render_image = np.concatenate(render_image, axis=1)
            if output_format == "images":
                if rendered_output_names[0] == "depth":
                    render_image = (render_image).astype(np.uint16)
                media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
            else:
                images.append(render_image)

    if output_format == "video":
        fps = len(images) / seconds
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Filename of the camera metadata to render.
    metadata_path: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename"] = "filename"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # How to save output data.
    output_format: Literal["images", "video"] = "images"
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None
    #angle for rotating novel views
    angle: float = 0.0

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test" if self.traj == "spiral" else "inference",
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds

        cameras = self.get_cameras(meta_data_path = self.metadata_path, cameras_save_path=self.camera_path_filename, angle = self.angle)

        _render_trajectory_video(
            pipeline,
            cameras,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
        )


    def get_cameras(self, meta_data_path = "meta_data.json", cameras_save_path="cameras.json", angle=0.0):
        # load meta data
        f = open(meta_data_path)
  
        meta = json.load(f)
        print(meta)

        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for i, frame in enumerate(meta["frames"]):

            print()
            print("frame", i)
            print(frame)
            print()
            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])
            
            # TODO: figure out what to put here
            #camtoworld = apply_something_to_camtoworld(camtoworld)
            
            

            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)


        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)
        
        ## moved this up here
        # camera_to_worlds[:, 0:3, 1:3] *= -1

        num_images = len(camera_to_worlds)

        camera_to_worlds_1 = camera_to_worlds[:(num_images//2), :, :]
        camera_to_worlds_2 = camera_to_worlds[(num_images//2):, :, :]

        t_1 = camera_to_worlds_1[:,0:3,3]
        t_2 = camera_to_worlds_2[:,0:3,3]

        t_i = (t_1 + t_2)/2
        t_i = np.vstack((t_i,t_i))

        rotmats_1 = camera_to_worlds_1[:,0:3,0:3]
        rotmats_2 = camera_to_worlds_2[:,0:3,0:3]

        rotmats_i1 = []
        rotmats_i2 = []
        rotmat_out_1 = get_rotmat(angle)
        rotmat_out_2 = get_rotmat(angle)
        for i in range (len(rotmats_1)):
            rotmat_1 = rotmats_1[i]
            rotmat_2 = rotmats_2[i]
            
            rotation_obj = R.from_matrix(np.stack((rotmat_1, rotmat_2)))

            slerp = Slerp(np.array([0, 1]), rotation_obj)
            rotmat_i = slerp(0.5)
            rotmat_i = rotmat_i.as_matrix()

            rotmat_i1 = np.dot(rotmat_out_1, rotmat_i)
            rotmat_i2 = np.dot(rotmat_out_2, rotmat_i)

            rotmats_i1.append(rotmat_i1)
            rotmats_i2.append(rotmat_i2)
        
        #change
        rotmats_i = np.vstack((rotmats_i1, rotmats_i2))

        camera_to_worlds = np.zeros((num_images, 4, 4))
        camera_to_worlds[:,3,3] = 1
        camera_to_worlds[:,0:3,0:3] = rotmats_i
        camera_to_worlds[:,0:3,3] = t_i
        
        print(camera_to_worlds.shape)
        print(camera_to_worlds[0])
         # TODO: figure out if this is needed
        
        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        camera_to_worlds = torch.tensor(camera_to_worlds).float()

        # if self.config.auto_orient:
        camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method="up",
            center_poses=False,
        )

        # CHANGE THIS!!!!!!
        height, width = 375, 1242#meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )
        
        
        camera_path = {}
        #camera_path['seconds'] = 5
        
        camera_list = []
        
        for i in range(len(meta["frames"])):
            camera_list.append(cameras.to_json(camera_idx=i))
        
        
        camera_path['cameras'] = camera_list
        
        #maybe uncomment idek
        # with open(os.path.join(save_loc,'transforms.json'),'w') as f:
        #     json.dump(d,f,indent=4)
        
        
        return cameras


if __name__ == "__main__":
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()



# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa



"""
Load a checkpoint, render a trajectory, and save as images.

╭─ arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                            │
│ --load-config PATH      Path to config YAML file. (required)                                                       │
│ --rendered-output-names STR [STR ...]                                                                              │
│                         Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis       │
│                         (default: rgb)                                                                             │
│ --traj {spiral,filename}                                                                                           │
│                         Trajectory to render. (default: filename)                                                    │
│ --downscale-factor INT  Scaling factor to apply to the camera image resolution. (default: 1)                        │
│ --output-path PATH      Name of the output file. (default: renders/output.mp4)                                     │
│ --seconds FLOAT         How long the video should be. (default: 5.0)                                               │
│ --output-format {images,video}                                                                                     │
│                         How to save output data. (default: images)                                                  │
│ --eval-num-rays-per-chunk {None}|INT                                                                               │
│                         Specifies number of rays per chunk during eval. (default: None)
"""


# python render.py --load-config ../outputs/calib_test/monosdf/2023-04-27_155630/config.yml --output-path ../outputs/calib_test/monosdf_novel_views/2023-04-27_155630.png

# 
#
#sbatch --wrap="python render.py --load-config ../outputs/calib_test/monosdf/2023-04-27_155630/config.yml --output-path ../outputs/calib_test/monosdf_novel_views/2023-04-27_155630.png" --time=1:0:0 --gpus=1 --ntasks=4 --mem-per-cpu=16G --job-name=novel_views

#