import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from Utils import set_logging_format, vis_disparity, set_seed
from image_common.image import load_l_r_images_undistorted
from image_common.utils import load_dict, get_l_r_image_fnames


# def parse_intrinsic_file(path: str):
#     with open(path, "r", encoding="utf-8") as f:
#         lines = [line.strip() for line in f.readlines() if line.strip()]
#     if len(lines) < 2:
#         raise ValueError("intrinsic_file must have 2 lines: flattened K and baseline")
#
#     k_vals = [float(v) for v in lines[0].split()]
#     if len(k_vals) != 9:
#         raise ValueError("first line in intrinsic_file must contain 9 numbers")
#
#     K = np.array(k_vals, dtype=np.float32).reshape(3, 3)
#     baseline = float(lines[1])
#     return K, baseline


# def list_pairs(left_dir: str, right_dir: str):
#     left_paths = {p.name: p for p in Path(left_dir).glob("*") if p.is_file()}
#     right_paths = {p.name: p for p in Path(right_dir).glob("*") if p.is_file()}
#     common = sorted(set(left_paths).intersection(right_paths))
#     return [(left_paths[name], right_paths[name], name) for name in common]

def get_K_baseline(calib_dict):
    K = calib_dict['K_l']
    tvec = calib_dict['tvec'].reshape(-1)
    baseline = float(np.linalg.norm(tvec))
    baseline = baseline / 1000.0
    return K, baseline


def main():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Run FoundationStereo on folders of left/right images")
    # parser.add_argument("--left_dir", required=True, type=str)
    # parser.add_argument("--right_dir", required=True, type=str)
    # parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth', type=str,
                        help='pretrained model path')
    # parser.add_argument("--intrinsic_file", type=str, default=None, help="Optional. If set, saves metric depth as .npy")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_imgs", type=int, default=5)
    args = parser.parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    NN_name = 'FoundationStereo'

    set_logging_format()
    set_seed(args.seed)
    torch.autograd.set_grad_enabled(False)

    root_dir = Path(__file__).resolve().parents[3]
    img_dir = root_dir / "dataset" / "depth" / "rgb"
    calib_dict_file = root_dir / "out" / "calib_data.npy"
    out_dir = root_dir / "out" / "stereo" / NN_name

    os.makedirs(out_dir, exist_ok=True)
    vis_dir = Path(out_dir) / "vis"
    disp_dir = Path(out_dir) / "disp"
    depth_dir = Path(out_dir) / "depth"
    vis_dir.mkdir(parents=True, exist_ok=True)
    disp_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    calib_dict = load_dict(calib_dict_file)
    K, baseline = get_K_baseline(calib_dict)

    cfg = OmegaConf.load(f"{os.path.dirname(args.ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for key, val in vars(args).items():
        cfg[key] = val
    cfg = OmegaConf.create(cfg)

    model = FoundationStereo(cfg)
    ckpt = torch.load(args.ckpt_dir, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    imgs_l, imgs_r = load_l_r_images_undistorted(
        calib_dict, img_dir, max_imgs=args.max_imgs
    )
    if len(imgs_l) == 0:
        raise RuntimeError("No stereo pairs found from the provided input arguments")

    logging.info("Found %d pairs", len(imgs_l))

    scale = args.scale

    for img_l, img_r in tqdm(zip(imgs_l, imgs_r), desc="inference"):
        img_number = img_l.number
        if scale is not None:
            img0 = img_l.get_small_img(scale=scale)
            img1 = img_r.get_small_img(scale=scale)

        h, w = img0.shape[:2]

        img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
        img0_t, img1_t = padder.pad(img0_t, img1_t)

        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0_t, img1_t, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(h, w)
        np.save(disp_dir / f"{img_number}_disp.npy", disp)
        print("File saved to:", disp_dir / f"{img_number}_disp.npy")
    #
        disp_vis = vis_disparity(disp)
        imageio.imwrite(vis_dir / f"{img_number}_vis.png", disp_vis)

        if K is not None and baseline is not None:
            k_scaled = K.copy()
            k_scaled[:2] *= args.scale
            depth = k_scaled[0, 0] * baseline / np.clip(disp, 1e-6, None)
            np.save(depth_dir / f"{img_number}_depth.npy", depth)


if __name__ == "__main__":
    main()
