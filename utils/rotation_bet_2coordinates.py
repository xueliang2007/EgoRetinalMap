"""
@brief: 已知单目的RGB和depth图像，通过在点云中分割出地面并拟合地面方程，来估计相机与地面之间的变换
"""

import os
import cv2
import glob
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rsci
from configs import cam_param_configs_sim


def get_ground_from_pcd(rgb_img_fn, depth_fn, cfg, vis):
    """

    :param rgb_img_fn: xx.png
    :param depth_fn:   xx.png
    :param cfg:
    :param vis:
    :return:
    """
    rgb = o3d.geometry.Image(o3d.io.read_image(rgb_img_fn))
    depth = o3d.geometry.Image(o3d.io.read_image(depth_fn))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0 / 25.5,
                                                              depth_trunc=50.0, convert_rgb_to_intensity=False)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(cfg["img_w"], cfg["img_h"], cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd_ds = pcd.voxel_down_sample(voxel_size=0.1)

    plane_model, inlier_idxs = pcd_ds.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=500)
    # [a, b, c, d] = plane_model
    # print(f"down sample, before: {len(pcd.points):d}, after: {len(pcd_ds.points):d}")
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    if vis:
        inlier_cloud = pcd_ds.select_by_index(inlier_idxs)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        outlier_cloud = pcd_ds.select_by_index(inlier_idxs, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, mesh], point_show_normal=True)

    return plane_model


def get_rotation_from_v2c_by_ground_plane_abcd(ground_plane_abcd, cfg):
    """
    前提：车辆坐标系：x向左，z向前，y向上
         相机坐标系：x向右，z向前，y向下
    已知：相机坐标系下的点云，并且已拟合出地面方程
    待求：从veh坐标系到相机坐标系的旋转矩阵
    """
    pvo = np.zeros((3, 1), dtype=np.float)
    pvo[1] = -ground_plane_abcd[3] / ground_plane_abcd[1]
    # print(f"pvo: {pvo.ravel()}")

    cam_k = np.array([[cfg["fx"], 0., cfg["cx"]],
                      [0., cfg["fy"], cfg["cy"]],
                      [0.,        0.,       1.]])
    img_ct = np.array([[cfg["cx"]], [cfg["img_h"]*0.8], [1]], dtype=np.float)
    p_ct = np.dot(np.linalg.inv(cam_k), img_ct)
    # print("p_ct'norm: ", p_ct.ravel())

    t = -ground_plane_abcd[3] / (np.dot(ground_plane_abcd[:3].reshape(1, 3), p_ct)[0][0])
    p_ct *= t
    # print("p_ct't: ", t)
    # print("p_ct: ", p_ct.ravel())

    dir_z = p_ct - pvo
    dir_z /= np.linalg.norm(dir_z)
    pz = pvo + dir_z
    # print("dir_z: ", dir_z.ravel())
    # print("pz: ", pz.ravel())

    dir_y = ground_plane_abcd[:3].reshape(3, 1)
    dir_y /= np.linalg.norm(dir_y)
    py = pvo - dir_y
    # print("dir_y: ", dir_y.ravel())
    # print("py: ", py.ravel())

    dir_x = np.cross(dir_y[:, 0], dir_z[:, 0]).reshape(3, 1)
    px = pvo - dir_x
    # print("dir_x: ", dir_x.ravel())
    # print("px: ", px.ravel())

    txyz = np.array([[0], [pvo[1]], [0]], dtype=np.float).reshape(1, 3)
    pvxyz = np.stack([px[:, 0], py[:, 0], pz[:, 0]]).reshape(3, 3)
    # print("pvxyz: \n", pvxyz)

    rot_mat = pvxyz - txyz
    # print("rot_mat: \n", rot_mat)

    be_rot = np.dot(rot_mat.T, rot_mat) - np.identity(3, dtype=np.float)
    err = np.linalg.norm(be_rot)
    if err > 1e-3:
        print("be_rot: \n", be_rot, "\nerr_norm: ", err)

    eulers = Rsci.from_matrix(rot_mat.T).as_euler("YXZ", degrees=True)
    # print("eulers: \n", eulers)
    return eulers


if __name__ == '__main__':
    cfgs = cam_param_configs_sim
    rgb_path = "../data/rgb"
    depth_path = "../data/depth"
    result_path = "../data/results"
    img_fns = sorted(glob.glob("%s/*.png" % rgb_path))

    for i, rgb_fn in enumerate(img_fns):
        basename = os.path.basename(rgb_fn)
        depth_fn = "%s/%s" % (depth_path, basename)
        gd_abcd = get_ground_from_pcd(rgb_fn, depth_fn, cfgs, vis=0)
        print(f"{i:d}, Plane equation: {gd_abcd[0]:.2f}x + {gd_abcd[1]:.2f}y + {gd_abcd[2]:.2f}z + {gd_abcd[3]:.2f} = 0")

        eulers_yxz = get_rotation_from_v2c_by_ground_plane_abcd(gd_abcd, cfgs)
        print(f"\teuler_v2c_in_deg: Y: {eulers_yxz[0]:.3f}, X: {eulers_yxz[1]:.3f}, Z: {eulers_yxz[2]:.3f}")

        img = cv2.imread(rgb_fn)
        txt1 = "gd plane: a:%.2f, b:%.2f, c:%.2f, d:%.2f" % tuple(gd_abcd.tolist())
        txt2 = "v2c euler_YXZ, Y:%.2f, X:%.2f, Z:%.2f" % tuple(eulers_yxz.tolist())
        cv2.putText(img, txt1, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
        cv2.putText(img, txt2, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
        cv2.imshow("img", img)
        cv2.waitKey()
        cv2.imwrite("%s/%s" % (result_path, basename), img)
        pass
