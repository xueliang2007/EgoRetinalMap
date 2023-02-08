import os
import cv2 as cv
import numpy as np
import open3d as o3d
from itertools import product
from matplotlib import pyplot as plt
from scipy.spatial.transform.rotation import Rotation as R_sci

from core.ego_pose import EgoPose
from core.uvs_transform import CoordTrans
from core.ego_data_convert import get_ground_from_disp

from utils import load_json, write_json, gather_results
from configs import get_cam_params, cam_param_self_sj


class EgoRetinalMapSelfData(EgoPose, CoordTrans):
    def __init__(self, cam_params, basenames_order):
        CoordTrans.__init__(self, cam_params["K"], cam_params["img_h"], cam_params["img_w"])
        EgoPose.__init__(self, cam_params["K"], cam_params["D"], cam_params["img_h"], cam_params["img_w"])

        self.remap_xy = None
        self.num_rhos = 500
        self.num_phis_half = 250
        self.rho_min = cam_params["rho_min"]
        self.rho_max = cam_params["rho_max"]

        self.phi_min = 1. / 6 * np.pi
        self.phi_max = 3. / 6 * np.pi
        self.basenames_order = basenames_order
        self.num_imgs = len(self.basenames_order)
        self.ground_method = cam_params["ground_method"]

        self.path_gy = cam_params["path_gy"]
        self.path_seg = cam_params["path_seg"]
        self.path_color = cam_params["path_color"]
        self.path_depth = cam_params["path_depth"]

        self.path_vis = cam_params.get("path_vis")
        self.path_traj_img = cam_params.get("path_traj_img")
        self.path_traj_ego = cam_params.get("path_traj_ego")
        self._path_results = cam_params["path_result"]

        self.pts_gd_veh = self.generate_polor_pts()
        self.fl_pose_infos = self.extract_flowlidar_pose(cam_params["fl_pose_txt"])

    def generate_polor_pts(self):
        phis = np.linspace(self.phi_min, self.phi_max, num=self.num_phis_half, endpoint=True)  # angle in rad
        tan_phis = np.tan(phis)
        coeffs = np.sqrt(1 + tan_phis * tan_phis)  # rho = coeffs * x, or e^{rho} = coeffs * x
        rhos = np.linspace(np.log(self.rho_min), np.log(self.rho_max), num=self.num_rhos, endpoint=True)
        rhos = np.exp(rhos)

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

        return np.array(xyzs_vehs).reshape((-1, 3))

    def extract_flowlidar_pose(self, pose_file):
        # image_name, qw, qx, qy, qz, tx, ty, tz
        if isinstance(pose_file, str):
            with open(pose_file, 'r') as f:
                lines = f.readlines()[1:]
        elif isinstance(pose_file, list):
            lines = pose_file

        n = len(lines)
        if n < 10:
            print("less pose record in this txt_log: %s\n" % os.path.basename(pose_file))
            return None

        fl_camh_list, fm_list, pts_list = [None] * n, [None] * n, [None] * n
        for i in range(n):
            line = lines[i][:-1]
            while line.find("  ") != -1:
                line.replace("  ", " ")

            info = line.split(' ')
            frame_name = info[0]
            t_xyz = [float(info[5]), float(info[6]), float(info[7])]
            quat_xyzw = [float(info[2]), float(info[3]), float(info[4]), float(info[1])]
            pxyz = -np.dot(R_sci.from_quat(quat_xyzw).as_matrix().T, t_xyz)
            fl_gd_a, fl_gd_b, fl_gd_c = float(info[13]), float(info[14]), float(info[15])
            fl_cam_h = 1. / np.sqrt(fl_gd_a * fl_gd_a + fl_gd_b * fl_gd_b + fl_gd_c * fl_gd_c)

            pts_list[i] = pxyz
            fl_camh_list[i] = fl_cam_h
            fm_list[i] = frame_name[:-4]
        return fm_list, pts_list, fl_camh_list

    def get_ground_abcd_from_img_depth(self, seg_mask, img_path, depth_path, vis):
        img = cv.imread(img_path)
        depth = np.load(depth_path)
        depth *= 0.001  # mm -> m
        disp = 1. / (depth + 1.e-5)
        Zs = depth.reshape((-1, 1))

        mesh_xy = np.meshgrid(range(self.img_w), range(self.img_h))
        uvs_mtx = np.stack((mesh_xy[0], mesh_xy[1]), axis=2)
        uvs = uvs_mtx.reshape((-1, 2))
        xyzs = np.dot(np.linalg.inv(self.cam_k), np.insert(uvs, 2, 1, 1).T).T * Zs  # Nx3
        rgbs = np.array([img[v, u][::-1] for u, v in uvs]).astype(np.float) / 255
        del uvs_mtx, mesh_xy, img, Zs

        mask_x = np.bitwise_and(xyzs[:, 0] > -4., xyzs[:, 0] < 4.)
        mask_y = xyzs[:, 1] > -2.
        mask_z = xyzs[:, 2] < 5.
        mask = np.bitwise_and(mask_x, np.bitwise_and(mask_y, mask_z))
        rgbs[np.bitwise_not(mask)] = [1, 0, 0]

        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyzs))
        pcd.colors = o3d.utility.Vector3dVector(rgbs)

        sample_idxs = np.arange(len(xyzs))[mask]
        pcd_crop = pcd.select_by_index(sample_idxs)
        del pcd, mask, mask_x, mask_y, mask_z

        num_bef = len(sample_idxs)
        pcd_ds = pcd_crop.voxel_down_sample(voxel_size=0.06)
        num_aft = len(pcd_ds.points)
        print(f"num of downsample, bef: {num_bef}, aft: {num_aft}")
        del pcd_crop

        gd_abcd, gd_idxs = pcd_ds.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)
        print("ground plane:", gd_abcd)

        if vis:
            pcd_gd = pcd_ds.select_by_index(gd_idxs, invert=False)
            pcd_ungd = pcd_ds.select_by_index(gd_idxs, invert=True)
            pcd_ungd.paint_uniform_color([1, 0, 0])
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            o3d.visualization.draw_geometries([pcd_gd, pcd_ungd, mesh])

        a, b, c, d = gd_abcd
        cam_h = np.fabs(d) / np.sqrt(a * a + b * b + c * c)
        return gd_abcd, cam_h, disp

    def get_ground_and_pose(self, basename, seg_mask, vis):
        bgr_fn = f"{self.path_color}/{basename}.jpg"
        depth_fn = f"{self.path_depth}/{basename}.npy"
        self.gd_abcd, self.cam_h, disp = self.get_ground_abcd_from_img_depth(seg_mask, bgr_fn, depth_fn, vis)

        eulers_xyz_v2c, txyz_v2c = self.get_ego_pose_v2c_without_instan_vel(self.gd_abcd)
        self.txyz_v2c = txyz_v2c
        self.eulers_xyz_v2c = eulers_xyz_v2c
        self.rot_mtx_v2c = R_sci.from_euler("XYZ", eulers_xyz_v2c, degrees=True).as_matrix()
        return disp

    def pts_veh2cam(self, pts_traj_v, pcd_vis=False):
        pts_traj_c = np.dot(self.rot_mtx_v2c, pts_traj_v.T).T + self.txyz_v2c
        pts_traj_c_norm = pts_traj_c / pts_traj_c[:, 2].reshape((-1, 1))
        uvs_traj_img = np.dot(self.cam_k, pts_traj_c_norm.T).T[:, :2]

        pts_gd_cam = np.dot(self.rot_mtx_v2c, self.pts_gd_veh.T).T + self.txyz_v2c
        pts_gd_cam_norm = pts_gd_cam / pts_gd_cam[:, 2][:, None]
        uvs_gd = np.dot(self.cam_k, pts_gd_cam_norm.T).T[:, :2]

        if pcd_vis:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            pcd_pts_veh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.pts_gd_veh))
            pcd_pts_cam = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_gd_cam))
            o3d.visualization.draw_geometries([pcd_pts_veh, mesh])
            o3d.visualization.draw_geometries([pcd_pts_cam, mesh])
        return uvs_gd, uvs_traj_img

    def get_ego_remap(self, uvs_img):
        self.remap_xy = np.zeros((self.num_rhos, self.num_phis_half * 2, 2), np.float32)
        for r, c in product(range(self.num_rhos), range(2 * self.num_phis_half)):
            self.remap_xy[self.num_rhos - 1 - r, c, :] = uvs_img[2 * self.num_phis_half * r + c]
        return

    def warp_for_gy(self, img, disp, seg, traj_iuvs, basename, show, save):
        """
        read like following:
            mask_ego = cv.imread(f"{self.path_gy}/{basename}.mask_ego.png", cv.IMREAD_UNCHANGED)
            ego_map = cv.imread(f"{self.path_gy}/{basename}.traj_ego.png", cv.IMREAD_UNCHANGED)
            disp_ego = cv.imread(f"{self.path_gy}/{basename}.disp_ego.tiff", cv.IMREAD_UNCHANGED)
        """
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        cv.polylines(mask, [traj_iuvs.astype(np.int)], isClosed=False, color=255, thickness=3)

        egos = []
        for img_i, border_color in zip([disp, mask, img, seg], [0, 0, [255, 255, 255], 0]):
            if len(img_i.shape) >= 2:
                ego_i = cv.remap(img_i, self.remap_xy[:, :, 0], self.remap_xy[:, :, 1], cv.INTER_LINEAR,
                                  borderMode=cv.BORDER_CONSTANT, borderValue=border_color)
                egos.append(ego_i)
        disp_ego, mask_ego, img_ego, seg_ego = egos

        if show:
            cv.imshow("ego_map", img_ego)
            cv.imshow("mask_ego", mask_ego)
            cv.waitKey()
            plt.imshow(disp_ego)
            plt.show()
            plt.close()

        if save:
            if np.isnan(disp_ego).any():
                return -1
            cv.imwrite(f"{self.path_gy}/{basename}.seg_ego.png", seg_ego)
            cv.imwrite(f"{self.path_gy}/{basename}.traj_ego.png", img_ego)
            cv.imwrite(f"{self.path_gy}/{basename}.mask_ego.png", mask_ego)
            cv.imwrite(f"{self.path_gy}/{basename}.disp_ego.tiff", disp_ego)

            # check
            if 1:
                img_ego_ip = cv.imread(f"{self.path_gy}/{basename}.traj_ego.png", cv.IMREAD_UNCHANGED)
                mask_ego_ip = cv.imread(f"{self.path_gy}/{basename}.mask_ego.png", cv.IMREAD_UNCHANGED)
                disp_ego_ip = cv.imread(f"{self.path_gy}/{basename}.disp_ego.tiff", cv.IMREAD_UNCHANGED)
                ego_map_err = np.sum(np.fabs(img_ego_ip - img_ego))
                mask_ego_err = np.sum(np.fabs(mask_ego_ip - mask_ego))
                disp_ego_err = np.sum(np.fabs(disp_ego_ip - disp_ego))
                assert ego_map_err + disp_ego_err + mask_ego_err < 1.e-3, "Save Failed!"

        return 0

    def vis_grids(self, img, disp, uvs_gd, traj_iuvs, basename, show, save):
        cv.polylines(img, [traj_iuvs.astype(np.int)], isClosed=False, color=(0, 0, 255), thickness=3)
        ego_map = cv.remap(img, self.remap_xy[:, :, 0], self.remap_xy[:, :, 1], cv.INTER_LINEAR,
                            borderMode=cv.BORDER_CONSTANT, borderValue=[255, 255, 255])

        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        for i in range(0, self.num_rhos, 5):
            idxs = [j for j in range(i*self.num_phis_half*2, (1+i)*self.num_phis_half*2, 5)]
            cv.polylines(img, [uvs_gd[idxs, :].astype(np.int32)], False, (255, 0, 0), 1)
        for i in range(0, self.num_phis_half*2, 5):
            idxs = [i+j*self.num_phis_half*2 for j in range(0, self.num_rhos, 5)]
            cv.polylines(img, [uvs_gd[idxs, :].astype(np.int32)], False, (0, 255, 0), 1)

        angx, angy, angz = self.eulers_xyz_v2c
        txt = f"{int(basename.split('_')[1]):d}, rho_max:{self.rho_max:.1f}, " \
              f"angx:{angx:.1f}, angy:{angy:.1f}, angz:{angz:.1f}"
        cv.rectangle(img, (0, 0), (self.img_w-1, 35), (250, 250, 250), -1)
        cv.putText(img, txt, (3, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

        img_rs = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        disp_rs = cv.resize(disp, (0, 0), fx=0.5, fy=0.5)
        if save:
            cv.imwrite(f"{self.path_vis}/{basename}.img_rs.png", img_rs)
            cv.imwrite(f"{self.path_vis}/{basename}.ego_map.png", ego_map)
            plt.imsave(f"{self.path_vis}/{basename}.disp_rs.jpg", disp_rs)

        if show:
            plt.imshow(disp_rs)
            plt.show()
            plt.close()
            cv.imshow("img_rs", img_rs)
            cv.imshow("ego_map", ego_map)
            cv.waitKey()
        return ego_map

    def update_json_result(self, traj_iuvs, traj_euvs, basename):
        traj_datas = {}
        traj_datas["traj_iuvs"] = traj_iuvs.tolist()
        traj_datas["traj_euvs"] = traj_euvs
        traj_datas["gd_abcd"] = self.gd_abcd.tolist()
        traj_datas["img_hw"] = [self.img_w, self.img_h]
        traj_datas["cam_k"] = self.cam_k.tolist()
        traj_json_fn = f"{self.path_traj_ego}/{basename}.ie.json"
        write_json(traj_datas, traj_json_fn)

    def run_single(self, img_idx):
        traj_iuvs = None
        basename = self.basenames_order[img_idx]
        traj_json_fn = f"{self.path_traj_img}/{basename}.json"
        if os.path.exists(traj_json_fn):
            traj_datas = load_json(traj_json_fn)
            traj_iuvs = np.array(traj_datas["traj_iuvs"])
            if len(traj_iuvs) < 8:
                print(f"\tDrop, i: {img_idx}, bsname: {basename}, num: {len(traj_iuvs)}")
                return
        if traj_iuvs is None:
            print(f"\tDrop, i: {img_idx}, bsname: {basename}, no trag datas")
            return

        seg = cv.imread(f"{self.path_seg}/{basename}")
        disp = self.get_ground(basename, seg, vis=False)
        if disp is None or self.cam_h is None or self.cam_h < 1.5:
            print(f"\tDrop for 'less pts in pcd', or 'disparity empty', i: {img_idx}, cam_h: {self.cam_h}")
            return

        if disp.dtype != np.float32:
            disp = disp.astype(np.float32)
        self.cam_pose_update(vel_instan_veh=None)
        angx, angy, angz = self.eulers_xyz_v2c
        if angx > 40 or abs(angy) > 12 or abs(abs(angz) - 180.) > 25:
            print(f"\tDrop, i: {img_idx}, bsname: {basename}, "
                  f"angx: {angx:.2f}, angy: {angy:.2f}, angz: {angz:.2f}, wrong ground and pose, may be indoor")
            status = 1
            if status:
                return

        uvs_gd = self.pts_veh2cam(pcd_vis=False)
        self.get_ego_remap(uvs_gd)

        img = cv.imread(f"{self.path_color}/{basename}")
        save_status = self.warp_for_gy(img, disp, seg, traj_iuvs, basename, show=False, save=True)
        if save_status == -1:
            print(f"\t Drop, i: {img_idx}, bsname: {basename}, NaN exits")
            return

        self.vis_grids(img, disp, uvs_gd, traj_iuvs, basename, show=0, save=True)
        print(f"\tDone, i: {img_idx}, bsname: {basename}, num_traj_filter: {len(traj_iuvs)}")

        traj_euvs = self.uvs_warp(self.remap_xy, traj_iuvs, method="i2e")
        self.update_json_result(traj_datas, traj_euvs, basename)
        pass


# def all_case():
#     cfgs = cam_param_self_sj
#     cases = sorted([d for d in os.listdir(cfgs["root"]) if os.path.isdir(f"{cfgs['root']}/{d}")])
#     for k, case in enumerate(cases):
#         cfgs["case"] = case
#         cam_params, basenames_order = get_cam_params(cfgs)
#         egr = EgoRetinalMapSelfData(cam_params, basenames_order)
#
#         if not os.path.exists(egr.path_gy):
#             os.makedirs(egr.path_gy, exist_ok=True)
#
#         for i in range(0, egr.num_imgs):
#             egr.run_single(img_idx=i)
#
#         print(f"k: {k}/{len(cases)}, case: {case} done!")
#
#     gather_results(cfgs["root"], dst_path=f"{cfgs['root']}/../all_case_results")


def simgle_image():
    cfgs = cam_param_self_sj
    cam_params, basenames_order = get_cam_params(cfgs)
    egr = EgoRetinalMapSelfData(cam_params, basenames_order)

    # img_path = f"/Users/xl/Desktop/ShaoJun/2022-09-15-15-53-03/color_sampled/1663067563.996764_00315.jpg"
    # depth_path = f"/Users/xl/Desktop/ShaoJun/2022-09-15-15-53-03/depth_sampled/1663067563.996764_00315.npy"
    img_idx = 0
    basename = egr.fl_pose_infos[0][img_idx]
    seg = np.zeros((egr.img_h, egr.img_w), dtype=np.uint8)
    disp = egr.get_ground_and_pose(basename, seg, vis=0)

    fl_cam_h = egr.fl_pose_infos[2][img_idx]
    zoom_scale = 3.
    zoom_scale_fl = egr.cam_h / fl_cam_h
    zoom_scale = 1./zoom_scale_fl

    traj_xyzs = np.array(egr.fl_pose_infos[1]).reshape((-1, 3)) * zoom_scale
    uvs_gd, traj_iuvs = egr.pts_veh2cam(pts_traj_v=traj_xyzs, pcd_vis=False)

    egr.get_ego_remap(uvs_gd)

    img = cv.imread(f"{egr.path_color}/{basename}.jpg")
    save_status = egr.warp_for_gy(img, disp, seg, traj_iuvs, basename, show=False, save=True)
    if save_status == -1:
        print(f"\t Drop, i: {img_idx}, bsname: {basename}, NaN exits")
        return

    egr.vis_grids(img, disp, uvs_gd, traj_iuvs, basename, show=1, save=True)
    print(f"\tDone, i: {img_idx}, bsname: {basename}, num_traj_filter: {len(traj_iuvs)}")

    traj_euvs = egr.uvs_warp(egr.remap_xy, traj_iuvs, method="i2e")
    egr.update_json_result(traj_iuvs, traj_euvs, basename)
    pass


if __name__ == '__main__':
    simgle_image()

