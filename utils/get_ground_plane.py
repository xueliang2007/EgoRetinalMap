"""
整理给GY
"""

import os
import cv2
import glob
import numpy as np
import open3d as o3d
from PIL import Image

from configs import cam_param_configs_sim


def get_ground_from_pcd(rgb_img_fn, depth_fn, cfg, vis, save_path, reproj_path):
    """
    :param rgb_img_fn: xx.jpg
    :param depth_fn:   xx.png
    :param cfg:
    :param vis:
    :return: [a, b, c, d]: ax+by+cz+d=0
    """
    rgb = o3d.geometry.Image(o3d.io.read_image(rgb_img_fn))
    depth = o3d.geometry.Image(o3d.io.read_image(depth_fn))  # ).astype(np.float)
    # depth = shicha / 10000
    # depth = Image.fromarray(shicha)

    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1./25.5, depth_trunc=50.0, convert_rgb_to_intensity=False)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1000, depth_trunc=50.0,
                                                              convert_rgb_to_intensity=False)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(cfg["img_w"], cfg["img_h"], cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd_ds = pcd.voxel_down_sample(voxel_size=0.6)

    plane_model, inlier_idxs = pcd_ds.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=500)
    # [a, b, c, d] = plane_model
    # print(f"down sample, before: {len(pcd.points):d}, after: {len(pcd_ds.points):d}")
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd_ds.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd_ds.select_by_index(inlier_idxs, invert=True)
    if vis:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, mesh], point_show_normal=True)

    if save_path:
        pcd_meg = inlier_cloud + outlier_cloud
        pcd_fn = f"{save_path}/pcd-{os.path.splitext(os.path.basename(rgb_img_fn))[0]}.ply"
        o3d.io.write_point_cloud(pcd_fn, pcd_meg)  # ply格式可以双击直接用Meshlab打开

    if reproj_path:
        pts = np.asarray(inlier_cloud.points)
        pts /= pts[:, 2].reshape((-1, 1))
        cam_k = np.array([[cfg["fx"], 0, cfg["cx"]], [0, cfg["fy"], cfg["cy"]], [0, 0, 1]], dtype=np.float)
        uvs = np.dot(cam_k, pts.T).T[:, :2]

        rad = 1
        weight = 0.7
        img = cv2.imread(rgb_img_fn)
        for u, v in uvs.astype(np.int):
            bgr1 = img[v - rad:v + rad + 1, u - rad:u + rad + 1, :]
            bgr2 = np.zeros_like(bgr1)
            bgr2[:, :, 2] = 255
            img[v - rad:v + rad + 1, u - rad:u + rad + 1, :] = bgr1 * weight + bgr2 * (1 - weight)

        reproj_fn = f"{save_path}/reproj-{os.path.splitext(os.path.basename(rgb_img_fn))[0]}.jpg"
        cv2.imwrite(reproj_fn, img)

    return plane_model


if __name__ == '__main__':
    cfgs = cam_param_configs_sim
    rgb_path = "../data/rgb"
    depth_path = "../data/depth"
    result_path = "../data/results"
    reproj_path = "../data/results"

    # img_fns = sorted(glob.glob("%s/*.png" % rgb_path))
    # for i, rgb_fn in enumerate(img_fns):
    #     basename = os.path.basename(rgb_fn)
    #     depth_fn = "%s/%s" % (depth_path, basename)
    #     gd_abcd = get_ground_from_pcd(rgb_fn, depth_fn, cfgs, vis=0, save_path=result_path, reproj_path=reproj_path)
    #     print(f"{i:d}, Plane equation: {gd_abcd[0]:.2f}x + {gd_abcd[1]:.2f}y + {gd_abcd[2]:.2f}z + {gd_abcd[3]:.2f} = 0")

    # rgb_path = "/Users/xl/Desktop/result/ori000067.png"
    # depth_path = "/Users/xl/Desktop/result/depth000067.png"
    rgb_path = "/Users/xl/Desktop/result/ori000026.png"
    depth_path = "/Users/xl/Desktop/result/depth000026.png"
    result_path = "/Users/xl/Desktop/result"
    reproj_path = "/Users/xl/Desktop/result"
    # result_path = ""
    # reproj_path = ""

    cfg = {
        "img_h": 720,
        "img_w": 1280,
        "fx": 960.,
        "fy": 960.,
        "cx": 640.,
        "cy": 360.,
        "D": np.zeros(5, dtype=np.float64),
    }

    rgb_fn = rgb_path
    depth_fn = depth_path
    gd_abcd = get_ground_from_pcd(rgb_fn, depth_fn, cfgs, vis=1, save_path=result_path, reproj_path=reproj_path)
    print(f"Plane equation: {gd_abcd[0]:.2f}x + {gd_abcd[1]:.2f}y + {gd_abcd[2]:.2f}z + {gd_abcd[3]:.2f} = 0")