import os
import cv2 as cv
import glob
import shutil
import numpy as np


def update_data_sj(color_path, depth_path, dst_path):
    for folder in ["color_ud", "depth_ud"]:
        if not os.path.exists(f"{dst_path}/{folder}"):
            os.makedirs(f"{dst_path}/{folder}")


    basenames = sorted([os.path.splitext(os.path.basename(fn))[0] for fn in glob.glob(f"{depth_path}/*.npy")])
    k = 0
    for name in basenames:
        src_color_fn = f"{color_path}/{name}.jpg"
        if not os.path.exists(src_color_fn):
            continue
        src_depth_fn = f"{depth_path}/{name}.npy"
        dst_depth_fn = f"{dst_path}/depth_ud/{name}{k:04d}.npy"
        dst_color_fn = f"{dst_path}/color_ud/{name}{k:04d}.jpg"
        shutil.copy(src_color_fn, dst_color_fn)
        shutil.copy(src_depth_fn, dst_depth_fn)
        k += 1


if __name__ == '__main__':
    update_data_sj(color_path="/Users/xl/Desktop/2022-09-15-15-53-03/color_image_raw",
                   depth_path="/Users/xl/Desktop/2022-09-15-15-53-03/aligned_depth_to_color_image_raw",
                   dst_path="/Users/xl/Desktop/2022-09-15-15-53-03")

# depth_fns = sorted(glob.glob(f"../data/depth_0/*.png"))
# for k, depth_fn in enumerate(depth_fns):
#     depth = cv.imread(depth_fn, cv.IMREAD_UNCHANGED)
#     depth_b, depth_g, depth_r = depth[:, :, 0], depth[:, :, 1], depth[:, :, 2]
#
#     cv.imshow("dep", depth)
#     cv.waitKey()
#     pass