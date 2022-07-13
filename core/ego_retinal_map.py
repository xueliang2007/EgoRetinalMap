import os
import cv2
import glob
import numpy as np
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation as Rsci
from configs import cam_param_configs_sim

cfgs = cam_param_configs_sim


class EgoRetinalMap:
    def __init__(self, pitch_init_v2c=None):
        cam_params, self.basenames_order = self.get_cam_params(cfgs, pitch_init_v2c)

        self.rho_min = 0.5
        self.rho_max = 25.

        self.num_rhos = 500
        self.num_phis_half = 250
        # self.num_rhos = 100
        # self.num_phis_half = 50

        self.phi_min = 1. / 6 * np.pi
        self.phi_max = 3. / 6 * np.pi
        self.num_imgs = len(self.basenames_order)

        self.gd_abcd = None
        self.txyz_v2c = None
        self.rot_mtx_v2c = None

        self.cam_k = cam_params["K"]
        self.cam_d = cam_params["D"]
        self.img_h = cam_params["img_h"]
        self.img_w = cam_params["img_w"]
        self.rpy_v2c = cam_params["rpy_v2c"]

        self.cam_k_o3d = o3d.camera.PinholeCameraIntrinsic(self.img_w, self.img_h, self.cam_k[0, 0],
                                                           self.cam_k[1, 1], self.cam_k[0, 2], self.cam_k[1, 2])

        self.path_color = cam_params["color_path"]
        self.path_depth = cam_params["depth_path"]
        self.path_results = cam_params["result_path"]

    def get_cam_params(self, cam_cfgs, pitch_v2c=None):
        # if pitch_v2c is not None:
        #     cam_cfgs["rpy_v2c"][1] = pitch_v2c
        fx = cam_cfgs["fx"]
        fy = cam_cfgs["fy"]
        cx = cam_cfgs["cx"]
        cy = cam_cfgs["cy"]
        cam_cfgs["K"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        cam_cfgs["depth_path"] = "%s/%s" % (cam_cfgs["root"], cam_cfgs["depth_path"])
        cam_cfgs["color_path"] = "%s/%s" % (cam_cfgs["root"], cam_cfgs["color_path"])
        cam_cfgs["result_path"] = "%s/%s" % (cam_cfgs["root"], cam_cfgs["result_path"])
        if not os.path.exists(cam_cfgs["result_path"]):
            os.makedirs(cam_cfgs["result_path"])

        img_fns = sorted(glob.glob("%s/*.%s" % (cam_cfgs["color_path"], cam_cfgs["img_suffix"])))
        basenames = [[]] * len(img_fns)
        for i, img_fn in enumerate(img_fns):
            basename_i = os.path.basename(img_fn)
            key = "0" * (6 - len(os.path.splitext(basename_i)[0])) + basename_i
            basenames[i] = [key, basename_i]
        basenames = sorted(basenames, key=lambda x: x[0])
        basenames_order = [bsname[1] for bsname in basenames]

        return cam_cfgs, basenames_order

    def get_ground_from_pcd(self, rgb_img_fn, depth_fn, vis):
        rgb = o3d.geometry.Image(o3d.io.read_image(rgb_img_fn))
        depth = o3d.geometry.Image(o3d.io.read_image(depth_fn))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0 / 25.5,
                                                                  depth_trunc=50.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.cam_k_o3d)

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

        a, b, c, d = plane_model
        cam_h = np.fabs(d) / np.sqrt(a * a + b * b + c * c)
        return plane_model, cam_h

    def get_remap_xy(self, uvs_img):
        remap_x = np.zeros((self.num_rhos, self.num_phis_half * 2), np.float32)
        remap_y = np.zeros((self.num_rhos, self.num_phis_half * 2), np.float32)
        for r in range(self.num_rhos):
            for c in range(2 * self.num_phis_half):
                u_img, v_img = uvs_img[2 * self.num_phis_half * r + c]
                remap_x[self.num_rhos - 1 - r, c] = u_img
                remap_y[self.num_rhos - 1 - r, c] = v_img

        return remap_x, remap_y

    def get_ego_retinal_map(self, img, remap_x, remap_y):
        ego_retinal_map = cv2.remap(img, remap_x, remap_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
        return ego_retinal_map

    def generate_polor_pts(self):
        phis = np.linspace(self.phi_min, self.phi_max, num=self.num_phis_half, endpoint=True)  # angle in rad
        tan_phis = np.tan(phis)
        coeffs = np.sqrt(1 + tan_phis * tan_phis)  # rho = coeffs * x, or e^{rho} = coeffs * x
        rhos = np.linspace(np.log(self.rho_min), np.log(self.rho_max), num=self.num_rhos, endpoint=True)
        rhos = np.exp(rhos)  # recover from log to exp
        # rhos = np.exp(rhos)  # log-polar's log

        xyzs_vehs = []
        for i, rho_i in enumerate(rhos):
            xs_pos = rho_i / coeffs
            zs_pos = xs_pos * tan_phis
            xs_neg = xs_pos[::-1] * -1.
            zs_neg = zs_pos[::-1]

            xs = np.hstack((xs_pos, xs_neg))
            ys = np.zeros(len(xs), dtype=np.float64)
            zs = np.hstack((zs_pos, zs_neg))

            xyzs_veh_rhoi = np.stack((xs, ys, zs), axis=1)
            xyzs_vehs.append(xyzs_veh_rhoi)
        return xyzs_vehs

    def get_zg_by_ego_vel(self, gd_abcd, vel_3d_instan):
        vel_3d_instan /= np.linalg.norm(vel_3d_instan)
        if vel_3d_instan.shape[0] != 3:
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

    def get_ego_pose_v2c(self, ground_plane_abcd, veh_vel_instan):
        """
        前提：车辆坐标系：x向左，z向前，y向上
             相机坐标系：x向右，z向前，y向下
        已知：相机坐标系下的点云，并且已拟合出地面方程
        待求：从veh坐标系到相机坐标系的旋转矩阵
        """
        # veh coordinate origin
        pvo = np.zeros((3, 1), dtype=np.float)
        pvo[1] = -ground_plane_abcd[3] / ground_plane_abcd[1]

        dir_z = self.get_zg_by_ego_vel(ground_plane_abcd, veh_vel_instan)
        pz = pvo + dir_z

        gd_norm = ground_plane_abcd[:3].reshape(3, 1)
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

        eulers_v2c = Rsci.from_matrix(rot_mat.T).as_euler("YXZ", degrees=True)
        txyz_v2c = -np.dot(rot_mat.T, txyz.T).T

        return eulers_v2c, txyz_v2c

    def cam_pose_update(self, bgr_fn, depth_fn, veh_vel_instan):
        """
        当瞬时速度向量的元素绝对值之和为零时，使用原有的标定参数计算位姿
        :param bgr_fn:
        :param depth_fn:
        :param vel_3d_instan:
        :return:
        """
        self.gd_abcd, cam_height = self.get_ground_from_pcd(bgr_fn, depth_fn, vis=False)
        if np.sum(np.abs(veh_vel_instan)) < 1.e-3:
            cam_xyz = np.array([[0., cam_height, 0.]], dtype=np.float64)
            self.rot_mtx_v2c = Rsci.from_euler("YXZ", [self.rpy_v2c[2], self.rpy_v2c[1], self.rpy_v2c[0]], degrees=True).as_matrix()
            self.txyz_v2c = -np.dot(self.rot_mtx_v2c, cam_xyz.T).T
            print(f"bd v2c t x: {self.txyz_v2c[0, 0]:.2f}, y: {self.txyz_v2c[0, 1]:.2f}, z: {self.txyz_v2c[0, 2]:.2f}")
            print(f"bd v2c euler x: {self.rpy_v2c[1]:.2f}, y: {self.rpy_v2c[0]:.2f}, z: {self.rpy_v2c[2]:.2f}")
        else:
            eulers_yxz, txyz = self.get_ego_pose_v2c(self.gd_abcd, veh_vel_instan)
            print(f"ego v2c t x: {txyz[0, 0]:.2f}, y: {txyz[0, 1]:.2f}, z: {txyz[0, 2]:.2f}")
            print(f"ego v2c euler x: {eulers_yxz[1]:.2f}, y: {eulers_yxz[0]:.2f}, z: {eulers_yxz[2]:.2f}")
            self.txyz_v2c = txyz
            self.rot_mtx_v2c = Rsci.from_euler("YXZ", eulers_yxz, degrees=True).as_matrix()
        return

    def pts_veh2cam(self, pts_gd_veh_list):
        pts_gd_veh = np.array(pts_gd_veh_list).reshape((-1, 3))
        pts_gd_cam = np.dot(self.rot_mtx_v2c, pts_gd_veh.T).T + self.txyz_v2c

        pts_gd_cam_norm = pts_gd_cam / pts_gd_cam[:, 2][:, None]
        uvs_gd = np.dot(self.cam_k, pts_gd_cam_norm.T).T[:, :2]

        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # pcd_pts_veh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_gd_veh))
        # pcd_pts_cam = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_gd_cam))
        # # o3d.visualization.draw_geometries([pcd_pts_veh, mesh])
        # o3d.visualization.draw_geometries([pcd_pts_cam, mesh])
        return uvs_gd, pts_gd_cam

    def vis_grids(self, img, ego_retinal_map, uvs_gd, basename, to_save):
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        for i in range(0, self.num_rhos, 5):
            idxs = [j for j in range(i*self.num_phis_half*2, (1+i)*self.num_phis_half*2, 5)]
            cv2.polylines(img, [uvs_gd[idxs, :].astype(np.int32)], False, (255, 0, 0), 1)
        for i in range(0, self.num_phis_half*2, 5):
            idxs = [i+j*self.num_phis_half*2 for j in range(0, self.num_rhos, 5)]
            cv2.polylines(img, [uvs_gd[idxs, :].astype(np.int32)], False, (0, 255, 0), 1)

        # for u, v in uvs_gd:
        #     cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255), -1)

        cv2.rectangle(img, (0, 0), (250, 20), (250, 250, 250), -1)
        txt = "%s, pitch: %.1f, rho_max: %.1f" % (basename, self.rpy_v2c[1], self.rho_max)
        cv2.putText(img, txt, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        if to_save:
            cv2.imwrite("%s/%s" % (self.path_results, basename.replace(".jpg", "-EgoCentric.jpg")), img)
            if ego_retinal_map is not None:
                cv2.imwrite("%s/%s" % (self.path_results, basename.replace(".jpg", "-EgoRetinalMap.jpg")), ego_retinal_map)

        return img

    def uv_from_ego_retinal_map_to_img(self, remap_x, remap_y, uv):
        u, v = uv
        ul0, vt0 = int(u), int(v)
        ur0, vb0 = ul0 + 1, vt0 + 1
        utl1, vtl1 = remap_x[vt0, ul0], remap_y[vt0, ul0]
        utr1, vtr1 = remap_x[vt0, ur0], remap_y[vt0, ur0]
        ubl1, vbl1 = remap_x[vb0, ul0], remap_y[vb0, ul0]
        ubr1, vbr1 = remap_x[vb0, ur0], remap_y[vb0, ur0]

        wu0, wu1 = u - ul0, ur0 - u
        wv0, wv1 = v - vt0, vb0 - v
        uop = (utl1 * wu0 + utr1 * wu1) * wv0 + (ubl1 * wu0 + ubr1 * wu1) * wv1
        vop = (vtl1 * wv0 + vbl1 * wv1) * wu0 + (vtr1 * wv0 + vbr1 * wv1) * wu1
        return [uop, vop]

    def uv_from_img_to_ego_retinal_map(self, remap_x, remap_y, uv):
        u, v = uv
        ul0, vt0 = int(u), int(v)
        ur0, vb0 = ul0 + 1, vt0 + 1
        utl1, vtl1 = remap_x[vt0, ul0], remap_y[vt0, ul0]
        utr1, vtr1 = remap_x[vt0, ur0], remap_y[vt0, ur0]
        ubl1, vbl1 = remap_x[vb0, ul0], remap_y[vb0, ul0]
        ubr1, vbr1 = remap_x[vb0, ur0], remap_y[vb0, ur0]

        wu0, wu1 = u - ul0, ur0 - u
        wv0, wv1 = v - vt0, vb0 - v
        uop = (utl1 * wu0 + utr1 * wu1) * wv0 + (ubl1 * wu0 + ubr1 * wu1) * wv1
        vop = (vtl1 * wv0 + vbl1 * wv1) * wu0 + (vtr1 * wv0 + vbr1 * wv1) * wu1
        return [uop, vop]

    def reproject_uvs_on_img_to_pts_on_ground(self, obj_uvs):
        """
        根据外参参数，通过目标点在图像上的投影恢复他在车辆坐标系下真实的3D坐标，其中假设3D坐标的Y为0
        """
        assert obj_uvs.shape[1] == 2  # Nx2

        K_inv = np.linalg.inv(self.cam_k)
        uv1s = np.insert(obj_uvs, 2, 1, axis=1)
        pts_prime = np.dot(K_inv, uv1s.T).T

        # solve: 's * (r01*x' + r11*y' + r21) = r01*tx + r11*ty + r21*tz' -> 's * left = right' -> s?
        right = np.dot(self.txyz_v2c, self.rot_mtx_v2c[:, 1])[0]
        lefts = np.dot(pts_prime, self.rot_mtx_v2c[:, 1])
        scales = right / lefts
        pts_cam = np.zeros_like(pts_prime)
        for i in range(3):
            pts_cam[:, i] = pts_prime[:, i] * scales
        pts_veh = np.dot(self.rot_mtx_v2c.T, (pts_cam - self.txyz_v2c).T).T

        return pts_veh

    def run_single(self, img_idx):
        basename = self.basenames_order[img_idx]
        bgr_fn = os.path.join(self.path_color, self.basenames_order[img_idx])
        depth_fn = os.path.join(self.path_depth, self.basenames_order[img_idx])

        # 如果是veh的运动方向，由于坐标系不同，所以需要将其绕Z轴旋转180度
        # veh_vel_instan = np.array([[0], [-0.2], [0.8]], dtype=np.float)
        cam_vel_instan = np.array([[0.], [-0.1], [0.8]], dtype=np.float)
        vel_instan = cam_vel_instan
        self.cam_pose_update(bgr_fn, depth_fn, vel_instan)

        pts_gd_veh_list = self.generate_polor_pts()
        uvs_gd, pts_gd_cam = self.pts_veh2cam(pts_gd_veh_list)

        # pts_veh = self.reproject_uvs_on_img_to_pts_on_ground(uvs_gd)

        img = cv2.imread(bgr_fn)
        remap_x, remap_y = self.get_remap_xy(uvs_gd)
        ego_retinal_map = self.get_ego_retinal_map(img, remap_x, remap_y)

        # uv_egr = [100.4, 100.6]
        # uv_img = self.uv_from_ego_retinal_map_to_img(remap_x, remap_y, uv_egr)
        img = self.vis_grids(img, ego_retinal_map, uvs_gd, basename, to_save=0)

        cv2.imshow("img", img)
        cv2.imshow("retinal", ego_retinal_map)
        cv2.waitKey()


if __name__ == '__main__':
    egr = EgoRetinalMap()
    egr.run_single(img_idx=1)
    # print(np.log(10.))
    # print(np.log(2.71828))

