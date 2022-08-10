import os
import cv2

import numpy as np
import open3d as o3d
from itertools import product
from matplotlib import pyplot as plt

from core.ego_pose import EgoPose
from core.uvs_transform import CoordTrans
from core.ego_data_convert import get_ground_from_disp

from utils import load_json, write_json, gather_results_gy
from configs import get_cam_params, cam_param_configs_sim, cam_param_ego_paper


class EgoRetinalMap(EgoPose, CoordTrans):
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
        self.path_color = cam_params["color_path"]
        self.path_depth = cam_params["depth_path"]

        self.path_vis = cam_params.get("path_vis")
        self.path_traj = cam_params.get("path_traj")
        self._path_results = cam_params["result_path"]

        self.pts_gd_veh = self.generate_polor_pts()

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

    def get_ground(self, basename, vis):
        bgr_fn = f"{self.path_color}/{basename}"
        if self.ground_method == "disp":
            disp_fn = f"{self.path_depth}/{basename}.disp.txt"
            self.gd_abcd, self.cam_h, disp = get_ground_from_disp(self.cam_k[0, 0], self.cam_k[1, 1],
                                                            self.cam_k[0, 2], self.cam_k[1, 2],
                                                            self.img_w, self.img_h, bgr_fn, disp_fn, vis)
        elif self.ground_method == "pcd":
            depth_fn = f"{self.path_depth}/{basename}"
            self.gd_abcd, self.cam_h, disp = self.get_ground_from_pcd(bgr_fn, depth_fn, vis)
        else:
            assert False, "Don't support!"

        return disp

    def pts_veh2cam(self, pcd_vis=False):
        pts_gd_cam = np.dot(self.rot_mtx_v2c, self.pts_gd_veh.T).T + self.txyz_v2c
        pts_gd_cam_norm = pts_gd_cam / pts_gd_cam[:, 2][:, None]
        uvs_gd = np.dot(self.cam_k, pts_gd_cam_norm.T).T[:, :2]

        if pcd_vis:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            pcd_pts_veh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.pts_gd_veh))
            pcd_pts_cam = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_gd_cam))
            o3d.visualization.draw_geometries([pcd_pts_veh, mesh])
            o3d.visualization.draw_geometries([pcd_pts_cam, mesh])
        return uvs_gd

    def get_ego_remap(self, uvs_img):
        self.remap_xy = np.zeros((self.num_rhos, self.num_phis_half * 2, 2), np.float32)
        for r, c in product(range(self.num_rhos), range(2 * self.num_phis_half)):
            self.remap_xy[self.num_rhos - 1 - r, c, :] = uvs_img[2 * self.num_phis_half * r + c]
        return

    def warp_for_gy(self, img, disp, traj_iuvs, basename, show, save):
        """
        read like following:
            mask_ego = cv2.imread(f"{self.path_gy}/{basename}.mask_ego.png", cv2.IMREAD_UNCHANGED)
            ego_map = cv2.imread(f"{self.path_gy}/{basename}.traj_ego.png", cv2.IMREAD_UNCHANGED)
            disp_ego = cv2.imread(f"{self.path_gy}/{basename}.disp_ego.tiff", cv2.IMREAD_UNCHANGED)
        """
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        cv2.polylines(mask, [traj_iuvs.astype(np.int)], isClosed=False, color=255, thickness=3)
        disp_ego = cv2.remap(disp, self.remap_xy[:, :, 0], self.remap_xy[:, :, 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_ego = cv2.remap(mask, self.remap_xy[:, :, 0], self.remap_xy[:, :, 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ego_map = cv2.remap(img, self.remap_xy[:, :, 0], self.remap_xy[:, :, 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
        if show:
            cv2.imshow("ego_map", ego_map)
            cv2.imshow("mask_ego", mask_ego)
            plt.imshow(disp_ego)
            plt.show()
            plt.close()

        if save:
            if np.isnan(disp_ego).any():
                return -1
            cv2.imwrite(f"{self.path_gy}/{basename}.traj_ego.png", ego_map)
            cv2.imwrite(f"{self.path_gy}/{basename}.mask_ego.png", mask_ego)
            cv2.imwrite(f"{self.path_gy}/{basename}.disp_ego.tiff", disp_ego)

            # check
            ego_map_ip = cv2.imread(f"{self.path_gy}/{basename}.traj_ego.png", cv2.IMREAD_UNCHANGED)
            mask_ego_ip = cv2.imread(f"{self.path_gy}/{basename}.mask_ego.png", cv2.IMREAD_UNCHANGED)
            disp_ego_ip = cv2.imread(f"{self.path_gy}/{basename}.disp_ego.tiff", cv2.IMREAD_UNCHANGED)
            ego_map_err = np.sum(np.fabs(ego_map_ip - ego_map))
            mask_ego_err = np.sum(np.fabs(mask_ego_ip - mask_ego))
            disp_ego_err = np.sum(np.fabs(disp_ego_ip - disp_ego))
            assert ego_map_err + disp_ego_err + mask_ego_err < 1.e-3, "Save Failed!"

        return 0

    def vis_grids(self, img, disp, uvs_gd, traj_iuvs, basename, show, save):
        cv2.polylines(img, [traj_iuvs.astype(np.int)], isClosed=False, color=(0, 0, 255), thickness=3)
        ego_map = cv2.remap(img, self.remap_xy[:, :, 0], self.remap_xy[:, :, 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])

        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        for i in range(0, self.num_rhos, 5):
            idxs = [j for j in range(i*self.num_phis_half*2, (1+i)*self.num_phis_half*2, 5)]
            cv2.polylines(img, [uvs_gd[idxs, :].astype(np.int32)], False, (255, 0, 0), 1)
        for i in range(0, self.num_phis_half*2, 5):
            idxs = [i+j*self.num_phis_half*2 for j in range(0, self.num_rhos, 5)]
            cv2.polylines(img, [uvs_gd[idxs, :].astype(np.int32)], False, (0, 255, 0), 1)

        cv2.rectangle(img, (0, 0), (self.img_w-1, 35), (250, 250, 250), -1)
        txt = "%s, rho_max: %.1f, angx: %.1f, angy: %.1f, angz: %.1f" % (basename, self.rho_max, self.eulers_xyz_v2c[0],
                                                                         self.eulers_xyz_v2c[1], self.eulers_xyz_v2c[2])
        cv2.putText(img, txt, (3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

        img_rs = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        disp_rs = cv2.resize(disp, (0, 0), fx=0.5, fy=0.5)
        if save:
            cv2.imwrite(f"{self.path_vis}/{basename}.img_rs.png", img_rs)
            cv2.imwrite(f"{self.path_vis}/{basename}.ego_map.png", ego_map)
            plt.imsave(f"{self.path_vis}/{basename}.disp_rs.jpg", disp_rs)

        if show:
            plt.imshow(disp_rs)
            plt.show()
            plt.close()
            cv2.imshow("img_rs", img_rs)
            cv2.imshow("ego_map", ego_map)
            cv2.waitKey()
        return ego_map

    def update_json_result(self, traj_datas, traj_euvs, traj_json_fn):
        traj_datas["traj_euvs"] = traj_euvs
        traj_datas["img_hw"] = [self.img_w, self.img_h]
        traj_datas["cam_k"] = self.cam_k.tolist()
        write_json(traj_datas, traj_json_fn.replace(".json", ".ie.json"))
        os.remove(traj_json_fn)

    def run_single(self, img_idx):
        traj_iuvs = None
        basename = self.basenames_order[img_idx]
        traj_json_fn = f"{self.path_traj}/{self.basenames_order[img_idx]}.json"
        if self.path_traj is not None and os.path.exists(traj_json_fn):
            traj_datas = load_json(traj_json_fn)
            traj_iuvs = np.array(traj_datas["traj_iuvs"])
            if len(traj_iuvs) < 10:
                print(f"\tdrop, i: {img_idx}, bsname: {basename}, num: {len(traj_iuvs)}")
                return
        if traj_iuvs is None:
            return

        disp = self.get_ground(basename, vis=False)
        if disp is None or self.cam_h is None or self.cam_h < 1.5:
            print(f"\tDrop for 'less pts in pcd', or 'disparity empty', i: {img_idx}, cam_h: {self.cam_h}")
            return

        if disp.dtype != np.float32:
            disp = disp.astype(np.float32)
        self.cam_pose_update(vel_instan_veh=None)
        angx, angy, angz = self.eulers_xyz_v2c
        if angx < 10 or abs(angy) > 10 or abs(abs(angz) - 180.) > 25:
            print(f"\tDrop for wrong ground and pose, may be indoor, i: {img_idx}, angx: {angx:.2f}, angy: {angy:.2f}, angz: {angz:.2f}")
            return

        uvs_gd = self.pts_veh2cam(pcd_vis=False)
        self.get_ego_remap(uvs_gd)

        img = cv2.imread(f"{self.path_color}/{basename}")
        save_status = self.warp_for_gy(img, disp, traj_iuvs, basename, show=False, save=True)
        if save_status:
            print(f"\t Drop for NaN exits, i: {img_idx}")
            return

        self.vis_grids(img, disp, uvs_gd, traj_iuvs, basename, show=0, save=True)
        print(f"\tdone_i: {img_idx}, bsname: {basename}, num_traj_filter: {len(traj_iuvs)}")

        traj_euvs = self.uvs_warp(self.remap_xy, traj_iuvs, method="i2e")
        self.update_json_result(traj_datas, traj_euvs, traj_json_fn)
        pass


def all_case():
    cfgs = cam_param_ego_paper
    cases = sorted([d for d in os.listdir(cfgs["root"]) if os.path.isdir(f"{cfgs['root']}/{d}")])
    for k, case in enumerate(cases):
        cfgs["case"] = case
        cam_params, basenames_order = get_cam_params(cfgs)
        egr = EgoRetinalMap(cam_params, basenames_order)
        for i in range(0, egr.num_imgs):
            egr.run_single(img_idx=i)

        print(f"k: {k}, case: {case} done!")

    gather_results_gy(cfgs["root"], dst_path=f"{cfgs['root']}/../all_case_results_for_gy_paper_datas")


if __name__ == '__main__':
    cfgs = [cam_param_configs_sim, cam_param_ego_paper][1]
    cam_params, basenames_order = get_cam_params(cfgs)
    egr = EgoRetinalMap(cam_params, basenames_order)
    egr.run_single(img_idx=2)

    # all_case()


