# Code adapted from the original SDF Studio repository:
# https://github.com/autonomousvision/sdfstudio/blob/master/scripts/render.py

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import torchvision.transforms as transforms
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
from nerfstudio.model_components.losses import compute_scale_and_shift

import os
import multiprocessing as mp

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
    reference_camera,
    reference_depth,
    scene_scale,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "images",
) -> None:
    """Helper function to create a video of the given trajectory.

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
    rgb_images = []
    depths = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    reference_camera = reference_camera.to(pipeline.device)

    #Getting depth estimates for a view of the scene that corresponds to a ground truth depth map
    gt_depth = torch.from_numpy(np.load(reference_depth))
    with torch.no_grad():
        output_depth = pipeline.model.get_outputs_for_camera_ray_bundle(reference_camera.generate_rays(camera_indices=0))["depth"].cpu()

    reference_crop = transforms.CenterCrop((output_depth.shape[0], output_depth.shape[1]))

    #Establishing scene scale and shift to be able to recover ground truth depth
    gt_depth = reference_crop(gt_depth)
    gt_depth = gt_depth / scene_scale
    gt_depth = gt_depth * 1000
    
    scale, shift = compute_scale_and_shift(output_depth[None, ..., 0], gt_depth[None, ...], gt_depth[None, ...] > 0.0)
    scale, shift = scale.item(), shift.item()

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    rgb_output_image_dir = output_image_dir / "rgb"
    depth_output_image_dir = output_image_dir / "depth"

    if output_format == "images":
        rgb_output_image_dir.mkdir(parents=True, exist_ok=True)
        depth_output_image_dir.mkdir(parents=True, exist_ok=True)
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            render_rgb_image = []
            render_depth_image = []
            output_rgb_image = outputs["rgb"].cpu().numpy()
            output_depth_image = outputs["depth"].cpu().numpy()
            render_rgb_image.append(output_rgb_image)
            render_depth_image.append(output_depth_image)
            render_rgb_image = np.concatenate(render_rgb_image, axis=1)
            render_depth_image = np.concatenate(render_depth_image, axis=1)
            render_depth_image = np.squeeze(render_depth_image, axis=2)
            render_depth_image = render_depth_image * scale + shift
            render_depth_image = np.clip(render_depth_image, 0, a_max=65535)
            render_depth_image = (render_depth_image).astype(np.uint16)
            rgb_images.append(render_rgb_image)
            depths.append(render_depth_image)

    if output_format == "images":
        print("saving images")
        for image_idx, render_image in enumerate(rgb_images):
            media.write_image(rgb_output_image_dir / f"{image_idx:05d}.png", render_image)
            print(f"saved image {image_idx}")
        for image_idx, render_image in enumerate(depths):
            media.write_image(depth_output_image_dir / f"{image_idx:05d}.png", render_image)
            print(f"saved depth image {image_idx}")
    if output_format == "video":
        fps = len(rgb_images) / seconds
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_filename, rgb_images, fps=fps)
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
    downscale_factor: float = 1
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

        cameras, reference_camera, reference_depth, scene_scale = self.get_cameras(meta_data_path = self.metadata_path, cameras_save_path=self.camera_path_filename, angle = self.angle)

        _render_trajectory_video(
            pipeline,
            cameras,
            output_filename=self.output_path,
            reference_camera=reference_camera,
            reference_depth=reference_depth,
            scene_scale=scene_scale,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
        )


    def get_cameras(self, meta_data_path = "meta_data.json", cameras_save_path="cameras.json", angle=0.0):
        """Get cameras for rendering.
        
        A camera includes the position and orientation of the camera, as well as the
        intrinsics of the camera.
        """
        # load meta data
        f = open(meta_data_path)
  
        meta = json.load(f)

        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        reference_camera = meta["frames"][0]
        
        scale_mat = np.linalg.inv(np.array(meta["worldtogt"]))
        scene_scale = scale_mat[0,0]

        #get metadata path folder
        meta_data_path_folder = Path(meta_data_path).parent
        reference_depth = meta_data_path_folder / reference_camera["mono_depth_path"]
        
        reference_cam_to_world = torch.tensor(reference_camera["camtoworld"]).unsqueeze(0)

        for i, frame in enumerate(meta["frames"]):

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

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

        num_images = len(camera_to_worlds)

        camera_to_worlds_1 = camera_to_worlds[:(num_images//2), :, :]
        camera_to_worlds_2 = camera_to_worlds[(num_images//2):, :, :]

        t_1 = camera_to_worlds_1[:,0:3,3]
        t_2 = camera_to_worlds_2[:,0:3,3]

        t_i = (t_1 + t_2)/2
        t_i = np.vstack((t_i,t_i))

        rotmats_1 = camera_to_worlds_1[:,0:3,0:3]
        rotmats_2 = camera_to_worlds_2[:,0:3,0:3]

        # Generating new camera poses according to the policy
        # described in the paper
        rotmats_i1 = []
        rotmats_i2 = []
        #getting rotation matrix for rotating novel views
        rotmat_out_1 = get_rotmat(angle)
        rotmat_out_2 = get_rotmat(-angle)
        for i in range (len(rotmats_1)):
            rotmat_1 = rotmats_1[i]
            rotmat_2 = rotmats_2[i]

            rotmat_i1 = np.dot(rotmat_out_1, rotmat_1)
            rotmat_i2 = np.dot(rotmat_out_2, rotmat_2)

            rotmats_i1.append(rotmat_i1)
            rotmats_i2.append(rotmat_i2)
        
        rotmats_i = np.vstack((rotmats_i1, rotmats_i2))

        camera_to_worlds = np.zeros((num_images, 4, 4))
        camera_to_worlds[:,3,3] = 1
        camera_to_worlds[:,0:3,0:3] = rotmats_i
        camera_to_worlds[:,0:3,3] = t_i
        
        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1
        reference_cam_to_world[:, 0:3, 1:3] *= -1

        camera_to_worlds = torch.tensor(camera_to_worlds).float()

        camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method="up",
            center_poses=False,
        )

        reference_cam_to_world, transform = camera_utils.auto_orient_and_center_poses(
            reference_cam_to_world,
            method="up",
            center_poses=False,
        )

        height, width = 375, 1242
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
        
        reference_camera = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=reference_cam_to_world[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )
        
        camera_path = {}
        
        camera_list = []
        
        for i in range(len(meta["frames"])):
            camera_list.append(cameras.to_json(camera_idx=i))
        
        camera_path['cameras'] = camera_list
        
        return cameras, reference_camera, reference_depth, scene_scale


if __name__ == "__main__":
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()

get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa