from pathlib import Path

import cv2
from matplotlib import pyplot as plt

from image_common.image import load_l_r_images_undistorted
from image_common.undistored_images import show_undistorted_images
from image_common.utils import get_l_r_image_fnames, load_dict

code_dir = Path(__file__).resolve().parent
root_dir = code_dir.parents[2]
img_dir = root_dir/ "dataset" / "depth" / "rgb"
calib_dict_file = root_dir / "out" / "calib_data.npy"
save_dir = root_dir / "out" / "depth_maps" / "FoundationStereo"
calib_dict = load_dict(calib_dict_file)
print(img_dir)
print(calib_dict_file)
max_imgs = 5

# images_l, images_r = get_l_r_image_fnames(img_dir, max_imgs)
# imgs_l, imgs_r = load_l_r_images_undistorted(
#         calib_dict, img_dir, max_imgs=max_imgs
#     )
show_undistorted_images(calib_dict, img_dir, max_imgs=max_imgs)