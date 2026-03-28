from __future__ import annotations

import argparse
import logging
import os
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

from code.image import load_l_r_images_rectified
from code.utils import load_dict, scale_intrinsics


def parse_args() -> argparse.Namespace:
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description="Run FoundationStereo on folders of rectified left/right images."
    )
    parser.add_argument(
        "--ckpt_dir",
        default=f"{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth",
        type=str,
        help="Pretrained model path",
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_imgs", type=int, default=5)
    parser.add_argument("--date", type=str, default="28032026")
    parser.add_argument(
        "--input_resolution",
        type=int,
        nargs=2,
        default=(3840, 2160),
        metavar=("W", "H"),
        help="Original calibration/image resolution used for intrinsics scaling.",
    )
    parser.add_argument(
        "--inference_resolution",
        type=int,
        nargs=2,
        default=(960, 540),
        metavar=("W", "H"),
        help="Resolution used for inference.",
    )
    parser.add_argument(
        "--save_depth_vis",
        action="store_true",
        default=True,
        help="Save depth visualization PNG files.",
    )
    parser.add_argument(
        "--min_disp",
        type=float,
        default=1e-6,
        help="Minimum disparity used to avoid division by zero.",
    )
    return parser.parse_args()


def build_paths(date: str) -> tuple[Path, Path, Path]:
    root_dir = Path(__file__).resolve().parents[3]
    img_dir = root_dir / "datasets" / f"dataset_{date}" / "stereo_4k_depth" / "rgb"
    calib_dict_file = root_dir / "out" / f"out_{date}" / "cameras_parameters" / "calib_data.npy"
    out_dir = root_dir / "out_estimation" / "stereo" / "FoundationStereo" / f"dataset_{date}"
    return img_dir, calib_dict_file, out_dir


def prepare_output_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "vis": out_dir / "vis",
        "disp": out_dir / "disp",
        "depth": out_dir / "depth",
        "depth_vis": out_dir / "depth_vis",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_calibration(calib_dict_file: Path, input_resolution: tuple[int, int], inference_resolution: tuple[int, int]) -> tuple[np.ndarray, float, dict]:
    calib_dict = load_dict(calib_dict_file)

    K = np.asarray(calib_dict["new_K_l"], dtype=np.float64)
    tvec = np.asarray(calib_dict["tvec"], dtype=np.float64).reshape(-1)

    baseline_m = float(np.linalg.norm(tvec)) / 1000.0
    K_scaled = scale_intrinsics(K, input_resolution, inference_resolution)

    return K_scaled, baseline_m, calib_dict


def load_model(args: argparse.Namespace) -> FoundationStereo:
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
    return model


def compute_disparity(
    model: FoundationStereo,
    img0: np.ndarray,
    img1: np.ndarray,
    valid_iters: int,
    hiera: int,
) -> np.ndarray:
    h, w = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.cuda.amp.autocast(True):
        if not hiera:
            disp = model.forward(img0_t, img1_t, iters=valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(
                img0_t,
                img1_t,
                iters=valid_iters,
                test_mode=True,
                small_ratio=0.5,
            )

    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(h, w)
    return disp.astype(np.float32)


def disparity_to_depth(
    disp: np.ndarray,
    fx: float,
    baseline_m: float,
    min_disp: float = 1e-6,
) -> np.ndarray:
    depth = np.full_like(disp, np.nan, dtype=np.float32)

    valid = np.isfinite(disp) & (disp > min_disp)
    depth[valid] = (fx * baseline_m) / disp[valid]

    return depth


def visualize_depth(depth: np.ndarray, max_percentile: float = 95.0) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    depth_valid = depth[valid]
    dmin = float(np.min(depth_valid))
    dmax = float(np.percentile(depth_valid, max_percentile))

    if dmax <= dmin:
        dmax = dmin + 1e-6

    depth_clipped = np.clip(depth, dmin, dmax)
    depth_norm = (depth_clipped - dmin) / (dmax - dmin)
    depth_norm[~valid] = 0.0

    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0

    return depth_color


def save_outputs(
    img_number: str,
    img_left: np.ndarray,
    disp: np.ndarray,
    depth: np.ndarray,
    out_dirs: dict[str, Path],
    save_depth_vis: bool = True,
) -> None:
    disp_path = out_dirs["disp"] / f"{img_number}_disp.npy"
    depth_path = out_dirs["depth"] / f"{img_number}_depth.npy"
    disp_vis_path = out_dirs["vis"] / f"{img_number}_disp_vis.png"
    preview_path = out_dirs["vis"] / f"{img_number}_preview.png"

    np.save(disp_path, disp)
    np.save(depth_path, depth)

    disp_vis = vis_disparity(disp)
    imageio.imwrite(disp_vis_path, disp_vis)

    preview = cv2.hconcat([img_left, disp_vis])
    imageio.imwrite(preview_path, preview)

    if save_depth_vis:
        depth_vis = visualize_depth(depth)
        depth_vis_path = out_dirs["depth_vis"] / f"{img_number}_depth_vis.png"
        imageio.imwrite(depth_vis_path, depth_vis)

    logging.info("Saved disparity: %s", disp_path)
    logging.info("Saved depth: %s", depth_path)


def main() -> None:
    args = parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    set_logging_format()
    set_seed(args.seed)
    torch.autograd.set_grad_enabled(False)

    img_dir, calib_dict_file, out_dir = build_paths(args.date)
    out_dirs = prepare_output_dirs(out_dir)

    logging.info("Image dir: %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir: %s", out_dir)

    K, baseline_m, calib_dict = load_calibration(
        calib_dict_file=calib_dict_file,
        input_resolution=tuple(args.input_resolution),
        inference_resolution=tuple(args.inference_resolution),
    )

    fx = float(K[0, 0])

    logging.info("Scaled fx: %.6f", fx)
    logging.info("Baseline: %.6f m", baseline_m)

    model = load_model(args)

    imgs_l, imgs_r = load_l_r_images_rectified(
        calib_dict,
        img_dir,
        max_imgs=args.max_imgs,
    )

    if len(imgs_l) == 0:
        raise RuntimeError("No stereo pairs found from the provided input arguments")

    logging.info("Found %d pairs", len(imgs_l))

    inference_resolution = tuple(args.inference_resolution)

    for img_l, img_r in tqdm(list(zip(imgs_l, imgs_r)), desc="inference"):
        img_number = img_l.get_image_number()

        img0 = img_l.get_resized_img(inference_resolution)
        img1 = img_r.get_resized_img(inference_resolution)

        disp = compute_disparity(
            model=model,
            img0=img0,
            img1=img1,
            valid_iters=args.valid_iters,
            hiera=args.hiera,
        )

        depth = disparity_to_depth(
            disp=disp,
            fx=fx,
            baseline_m=baseline_m,
            min_disp=args.min_disp,
        )

        save_outputs(
            img_number=img_number,
            img_left=img0,
            disp=disp,
            depth=depth,
            out_dirs=out_dirs,
            save_depth_vis=args.save_depth_vis,
        )


if __name__ == "__main__":
    main()