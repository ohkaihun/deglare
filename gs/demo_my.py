#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
# from light_model.utils import gen_mask
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussian_renderer import GaussianModel
from utils.sh_utils import eval_sh
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math
from PIL import Image

def prune_by_gradients(viewpoint_cameras, pc : GaussianModel, pipe, bg_color : torch.Tensor, dataset,scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    frame_idx = 0

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    shs=pc.get_features
    shs = torch.tensor(shs, requires_grad=True)
    gaussian_grads = torch.zeros(shs.shape[0], device=shs.device)
    for viewpoint_camera in viewpoint_cameras:



        # light_mask = torch.zeros_like(viewpoint_camera.original_image.cuda())
        # light_mask = (light_mask != 0)

        mask_PIL = Image.open(os.path.join(dataset.source_path, 'mask_large', viewpoint_camera.image_name))
        mask = torch.from_numpy(np.array(mask_PIL)).to("cuda") / 255.0
        if mask.dim() == 2:  # 如果是二维张量
            # 在最后一个维度增加一个维度，然后复制到三通道
            mask = mask.unsqueeze(2).repeat(1, 1, 3)  # 形状变为 (高度, 宽度, 3)
        mask = mask.permute(2, 0, 1)

        # for mask_idx in range(3):
        #     mask_np = gen_mask(light_centers[mask_idx * 3, 0].cpu().numpy(),
        #                        light_centers[mask_idx * 3, 1].cpu().numpy(), viewpoint_camera.image_height, viewpoint_camera.image_width,
        #                        5357, mask_idx)
        #     mask = torch.tensor(mask_np, dtype=torch.float32).to(
        #         torch.device('cuda')).permute(2, 0, 1) / 255.
        #     mask = (mask == 1.)
        #     light_mask = light_mask | mask
        # light_mask=~light_mask

        # light_center = centers[viewpoint_camera.colmap_id - 1]
        # light_mask_np = gen_mask(light_center[0].cpu().numpy(), light_center[1].cpu().numpy(), viewpoint_camera.image_height, viewpoint_camera.image_width, 0)
        # light_mask_float = torch.tensor(light_mask_np, dtype=torch.float32).to(torch.device('cuda')).permute(2, 0,
        #                                                                                                      1) / 255.
        # light_mask = light_mask_float == 1.
        # fit over light

        print(1)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            antialiasing=pipe.antialiasing
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)


        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.



        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.

        colors_precomp = None





        # Rasterize visible Gaussians to image, obtain their radii (on screen).

        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # Apply exposure to rendered image (training only)
        if use_trained_exp:
            exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0,
                                                                                                     1) + exposure[:3,
                                                                                                          3, None, None]

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rendered_image = rendered_image.clamp(0, 1)



        frame_idx += 1
        # pseudo_loss = ((rendered_image.detach() + 1 - rendered_image)**2).mean()

        target = rendered_image* mask
        pseudo_loss = target.mean()
        pseudo_loss.backward(retain_graph=True)
        # print(colors.grad.shape)
        gaussian_grads += (shs.grad[:, 0]).norm(dim=[1])
        shs.grad.zero_()




    mask = gaussian_grads > 0
    print("Total splats", len(gaussian_grads))
    print("Pruned", (~mask).sum(), "splats")
    print("Remaining", mask.sum(), "splats")
    pc.prune_pointsby2dmask(~mask)
    return pc, mask

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_novel_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "novel_renders")

    makedirs(render_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]


        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]


        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))



def render_sets(dataset : ModelParams,opt, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    # with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    checkpoint=r'C:\Users\Mayn\work\2dgs_water\gaussian-splatting\output\0312_data1_light_gs_del_3\chkpnt7000.pth'
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    # gaussians.training_setup(opt)


    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    centers = torch.tensor(np.load(os.path.join(dataset.source_path, 'center_3light.npy')), dtype=torch.float32).to(torch.device('cuda'))



    pc,mask=prune_by_gradients(scene.getTrainCameras(), gaussians, pipeline, background,centers, use_trained_exp=dataset.train_test_exp,
                                separate_sh=SPARSE_ADAM_AVAILABLE)



    if not skip_train:
         render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), pc, pipeline, background, dataset.train_test_exp, separate_sh)

    if not skip_test:
         render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

    render_novel_set(dataset.model_path, "novel", scene.loaded_iter, scene.getNovelCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args),op.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)