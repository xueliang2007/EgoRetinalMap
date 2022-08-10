import os
import cv2
import glob
import numpy as np
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation as Rsci

from configs import cam_param_configs_sim


class EgoPose:
    def __init__(self, cam_K, cam_D, img_h, img_w):
        self.cam_h = None
        self.gd_abcd = None
        self.txyz_v2c = None
        self.rot_mtx_v2c = None
        self.eulers_xyz_v2c = None

        self.cam_k = cam_K
        self.cam_d = cam_D
        self.img_h = img_h
        self.img_w = img_w

    def get_ground_from_pcd(self, rgb_img_fn, depth_fn, vis):
        rgb = o3d.geometry.Image(o3d.io.read_image(rgb_img_fn))
        depth = o3d.geometry.Image(o3d.io.read_image(depth_fn))

        cam_k_o3d = o3d.camera.PinholeCameraIntrinsic(self.img_w, self.img_h,
                                                      self.cam_k[0, 0], self.cam_k[1, 1],
                                                      self.cam_k[0, 2], self.cam_k[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0 / 25.5,
                                                                  depth_trunc=50.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_k_o3d)

        pcd_ds = pcd.voxel_down_sample(voxel_size=0.1)
        plane_model, inlier_idxs = pcd_ds.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=500)

        if vis:
            inlier_cloud = pcd_ds.select_by_index(inlier_idxs)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])

            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            outlier_cloud = pcd_ds.select_by_index(inlier_idxs, invert=True)
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, mesh], point_show_normal=True)

        a, b, c, d = plane_model
        cam_h = np.fabs(d) / np.sqrt(a * a + b * b + c * c)
        disp = 1. / depth
        return plane_model, cam_h, disp

    def get_zg_by_ego_vel(self, gd_abcd, vel_3d_instan):
        vel_3d_instan /= np.linalg.norm(vel_3d_instan)
        if vel_3d_instan.shape[0] != 3 or len(vel_3d_instan.shape) == 1:
            vel_3d_instan = vel_3d_instan.reshape(3, 1)

        gd_norm = np.array([[gd_abcd[0]], [gd_abcd[1]], [gd_abcd[2]]], dtype=np.float)
        zg = vel_3d_instan - np.dot(gd_norm.T, vel_3d_instan) * vel_3d_instan
        zg /= np.linalg.norm(zg)
        return zg

    def get_yg(self, zg, gd_norm):
        """
        1. For in the same plane, so vector yg = a * zg + gd_norm
        2. yg _|_ zg, so [yg_x, yg_y, yg_z] * [zg_x, zg_y, zg_z]^T = 0
        """
        pass
        gd_norm = gd_norm.reshape(3, 1)
        zg = zg.reshape(3, 1)

        a = -np.dot(zg.T, gd_norm) / np.linalg.norm(zg)

        yg = a[0, 0] * zg + gd_norm
        yg /= np.linalg.norm(yg)
        return yg

    def get_ego_pose_v2c_with_instan_vel(self, gd_abcd, veh_vel_instan):
        """
        前提：车辆坐标系：x向左，z向前，y向上
             相机坐标系：x向右，z向前，y向下
        已知：相机坐标系下的点云，并且已拟合出地面方程
        待求：从veh坐标系到相机坐标系的旋转矩阵
        """
        pvo = np.zeros((3, 1), dtype=np.float)
        pvo[1] = -gd_abcd[3] / gd_abcd[1]

        dir_z = self.get_zg_by_ego_vel(gd_abcd, veh_vel_instan)
        pz = pvo + dir_z

        gd_norm = gd_abcd[:3].reshape(3, 1)
        dir_y = self.get_yg(dir_z, gd_norm)
        py = pvo - dir_y

        dir_x = np.cross(dir_y[:, 0], dir_z[:, 0]).reshape(3, 1)
        px = pvo - dir_x

        txyz = np.array([[0], [pvo[1]], [0]], dtype=np.float).reshape(1, 3)
        pvxyz = np.stack([px[:, 0], py[:, 0], pz[:, 0]]).reshape(3, 3)
        rot_mat = pvxyz - txyz

        be_rot = np.dot(rot_mat.T, rot_mat) - np.identity(3, dtype=np.float)
        err = np.linalg.norm(be_rot)
        if err > 1e-3:
            print("be_rot: \n", be_rot, "\nerr_norm: ", err)

        txyz_v2c = -np.dot(rot_mat.T, txyz.T).T

        eulers_v2c = Rsci.from_matrix(rot_mat.T).as_euler("XYZ", degrees=True)

        return eulers_v2c, txyz_v2c

    def get_ego_pose_v2c_without_instan_vel(self, gd_abcd):
        """
        前提：车辆坐标系：x向左，z向前，y向上
             相机坐标系：x向右，z向前，y向下
        已知：相机坐标系下的点云，并且已拟合出地面方程
        待求：从veh坐标系到相机坐标系的旋转矩阵
        """
        pvo = np.zeros((3, 1), dtype=np.float)
        pvo[1] = -gd_abcd[3] / gd_abcd[1]

        img_ct = np.array([[self.cam_k[0, 2]], [self.img_h * 0.8], [1]], dtype=np.float)
        p_ct = np.dot(np.linalg.inv(self.cam_k), img_ct)

        t = -gd_abcd[3] / (np.dot(gd_abcd[:3].reshape(1, 3), p_ct)[0][0])
        p_ct *= t

        dir_z = p_ct - pvo
        dir_z /= np.linalg.norm(dir_z)
        pz = pvo + dir_z

        dir_y = gd_abcd[:3].reshape(3, 1)
        dir_y /= np.linalg.norm(dir_y)
        py = pvo - dir_y

        dir_x = np.cross(dir_y[:, 0], dir_z[:, 0]).reshape(3, 1)
        px = pvo - dir_x

        txyz = np.array([[0], [pvo[1]], [0]], dtype=np.float).reshape(1, 3)
        pvxyz = np.stack([px[:, 0], py[:, 0], pz[:, 0]]).reshape(3, 3)
        rot_mat = pvxyz - txyz

        be_rot = np.dot(rot_mat.T, rot_mat) - np.identity(3, dtype=np.float)
        err = np.linalg.norm(be_rot)
        if err > 1e-3:
            assert False, f"be_rot: \n{be_rot}\nerr_norm: {err:.5f}"

        eulers_v2c = Rsci.from_matrix(rot_mat.T).as_euler("XYZ", degrees=True)
        txyz_v2c = -np.dot(rot_mat.T, txyz.T).T
        return eulers_v2c, txyz_v2c

    def cam_pose_update(self, vel_instan_veh):
        if vel_instan_veh is None:
            eulers_xyz_v2c, txyz_v2c = self.get_ego_pose_v2c_without_instan_vel(self.gd_abcd)
        else:
            # 如果是veh的运动方向，由于坐标系不同，所以需要将其绕Z轴旋转180度
            eulers_xyz_v2c, txyz_v2c = self.get_ego_pose_v2c_with_instan_vel(self.gd_abcd, vel_instan_veh)

        self.txyz_v2c = txyz_v2c
        self.eulers_xyz_v2c = eulers_xyz_v2c
        self.rot_mtx_v2c = Rsci.from_euler("XYZ", eulers_xyz_v2c, degrees=True).as_matrix()
        return


def demo():
    cfgs = cam_param_configs_sim
    cfgs["K"] = np.array([[cfgs["fx"], 0, cfgs["cx"]],
                          [0, cfgs["fy"], cfgs["cy"]],
                          [0, 0, 1]], dtype=np.float64)

    bgr_fn = "../data/rgb/0.png"
    depth_fn = "../data/depth/0.png"
    ep = EgoPose(cfgs["K"], cfgs["D"], cfgs["img_h"], cfgs["img_w"])
    ep.cam_pose_update(bgr_fn, depth_fn, vel_instan_veh=None, pcd_vis=False)
    ep.cam_pose_update(bgr_fn, depth_fn, vel_instan_veh=np.array([0.2, 0, 1]), pcd_vis=False)
    pass


# if __name__ == '__main__':
#     demo()


