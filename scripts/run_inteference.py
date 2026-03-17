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


from calibration.image import get_undistort_functions, load_l_r_images_undistorted, load_l_r_images_rectified
from utils import load_dict, scale_intrinsics



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
    parser.add_argument("--hiera", type=int, default=0)  # maybe 1 fo 23-51-11
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_imgs", type=int, default=5)
    args = parser.parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    NN_name = 'FoundationStereo'

    set_logging_format()
    set_seed(args.seed)
    torch.autograd.set_grad_enabled(False)

    root_dir = Path(__file__).resolve().parents[3]
    img_dir = root_dir / "DiplomaThesis_git" / "dataset_11032026" / "stereo_4k_depth" / "rgb"
    calib_dict_file = root_dir / "DiplomaThesis_git" / "out" / "cameras_parameters" / "calib_data.npy"
    out_dir = root_dir / "out_estimation" / "stereo" / NN_name

    os.makedirs(out_dir, exist_ok=True)
    vis_dir = Path(out_dir) / "vis"
    disp_dir = Path(out_dir) / "disp"
    depth_dir = Path(out_dir) / "depth"
    vis_dir.mkdir(parents=True, exist_ok=True)
    disp_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    calib_dict = load_dict(calib_dict_file)
    K = calib_dict['K_l']
    tvec = calib_dict['tvec'].reshape(-1)
    baseline = float(np.linalg.norm(tvec))
    baseline = baseline / 1000.0

    print(img_dir)

    smaller_resolution = (960, 540)

    K = scale_intrinsics(K, (3840, 2160), smaller_resolution)

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

    imgs_l, imgs_r = load_l_r_images_rectified(
        calib_dict, img_dir, max_imgs=args.max_imgs
    )
    if len(imgs_l) == 0:
        raise RuntimeError("No stereo pairs found from the provided input arguments")

    logging.info("Found %d pairs", len(imgs_l))

    for img_l, img_r in tqdm(zip(imgs_l, imgs_r), desc="inference"):
        img_number = img_l.get_image_number()
        img0 = img_l.get_resized_img(smaller_resolution)
        img1 = img_r.get_resized_img(smaller_resolution)

        h, w = img0.shape[:2]
        print(img0.shape)

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

        vis = cv2.hconcat([img0, disp_vis])
        # cv2.imshow("vis", vis)
        # cv2.waitKey(0)

        if K is not None and baseline is not None:
            depth = K[0, 0] * baseline / np.clip(disp, 1e-6, None)


if __name__ == "__main__":
    main()
