import os
import cv2
# import glob
# import json
import numpy as np
import open3d as o3d
# from scipy.spatial.transform.rotation import Rotation as Rsci

from core.ego_pose import EgoPose
from core.uvs_transform import CoordTrans
from core.ego_data_convert import get_ground_from_disp

from utils import load_json, write_json
from configs import get_cam_params, cam_param_configs_sim, cam_param_ego_paper


class EgoRetinalMap(EgoPose, CoordTrans):
    def __init__(self, cam_params, basenames_order):
        CoordTrans.__init__(self, cam_params["K"])
        EgoPose.__init__(self, cam_params["K"], cam_params["D"], cam_params["img_h"], cam_params["img_w"])

        self.remap_xy = None
        self.rho_min = 0.5
        self.rho_max = 25.
        self.num_rhos = 500
        self.num_phis_half = 250

        self.phi_min = 1. / 6 * np.pi
        self.phi_max = 3. / 6 * np.pi
        self.basenames_order = basenames_order
        self.num_imgs = len(self.basenames_order)
        self.ground_method = cam_params["ground_method"]

        self.path_color = cam_params["color_path"]
        self.path_depth = cam_params["depth_path"]
        self.path_results = cam_params["result_path"]
        self.path_traj = cam_params.get("path_traj")

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

    def get_ground(self, bgr_fn, disp_depth_fn, vis):
        if self.ground_method == "disp":
            self.gd_abcd, self.cam_h = get_ground_from_disp(self.cam_k[0, 0], self.cam_k[1, 2],
                                                            self.cam_k[0, 2], self.cam_k[1, 2],
                                                            self.img_w, self.img_h, bgr_fn, disp_depth_fn, vis)
        elif self.ground_method == "pcd":
            self.gd_abcd, self.cam_h = self.get_ground_from_pcd(bgr_fn, disp_depth_fn, vis)
        else:
            assert False, "Don't support!"

        a, b, c, d = self.gd_abcd
        print(f"\tground_plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0, cam_h: {self.cam_h}")
        return

    def pts_veh2cam(self, vis=False):
        pts_gd_cam = np.dot(self.rot_mtx_v2c, self.pts_gd_veh.T).T + self.txyz_v2c
        pts_gd_cam_norm = pts_gd_cam / pts_gd_cam[:, 2][:, None]
        uvs_gd = np.dot(self.cam_k, pts_gd_cam_norm.T).T[:, :2]

        if vis:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            pcd_pts_veh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.pts_gd_veh))
            pcd_pts_cam = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_gd_cam))
            o3d.visualization.draw_geometries([pcd_pts_veh, mesh])
            o3d.visualization.draw_geometries([pcd_pts_cam, mesh])
        return uvs_gd

    def get_ego_retinal_map(self, img, uvs_img):
        self.remap_xy = np.zeros((self.num_rhos, self.num_phis_half * 2, 2), np.float32)
        for r in range(self.num_rhos):
            for c in range(2 * self.num_phis_half):
                self.remap_xy[self.num_rhos - 1 - r, c, :] = uvs_img[2 * self.num_phis_half * r + c]

        ego_retinal_map = cv2.remap(img, self.remap_xy[:, :, 0], self.remap_xy[:, :, 1], cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
        return ego_retinal_map

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

        cv2.rectangle(img, (0, 0), (400, 20), (250, 250, 250), -1)
        txt = "%s, rho_max: %.1f, angx: %.1f, angy: %.1f, angz: %.1f" % (basename, self.rho_max, self.eulers_yxz_v2c[1],
                                                                         self.eulers_yxz_v2c[0], self.eulers_yxz_v2c[2])
        cv2.putText(img, txt, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        if to_save:
            cv2.imwrite("%s/%s" % (self.path_results, basename.replace(".jpg", "-EgoCentric.jpg")), img)
            if ego_retinal_map is not None:
                cv2.imwrite("%s/%s" % (self.path_results, basename.replace(".jpg", "-EgoRetinalMap.jpg")), ego_retinal_map)

        return img

    def run_single(self, img_idx):
        basename = self.basenames_order[img_idx]
        bgr_fn = f"{self.path_color}/{self.basenames_order[img_idx]}"
        depth_fn = f"{self.path_depth}/{self.basenames_order[img_idx]}"

        self.get_ground(bgr_fn, depth_fn, vis=False)
        self.cam_pose_update(vel_instan_veh=None)

        img = cv2.imread(bgr_fn)
        uvs_gd = self.pts_veh2cam(vis=1)
        ego_retinal_map = self.get_ego_retinal_map(img, uvs_gd)
        img = self.vis_grids(img, ego_retinal_map, uvs_gd, basename, to_save=1)

        # traj_json_fn = f"{self.path_traj}/{self.basenames_order[img_idx]}.json"
        # if self.path_traj is not None and os.path.exists(traj_json_fn):
        #     traj_datas = load_json(traj_json_fn)
        #     traj_iuvs = traj_datas["traj_iuvs"]
        #     traj_euvs = self.uvs_img2ego(self.remap_xy, traj_iuvs)
        #     plot = 1
        #     if plot:
        #         # cv2.polylines(img, [tr_gd.astype(np.int)], isClosed=False, color=(0, 0, 255), thickness=2)
        #         # cv2.imwrite(f"{self.traj_vis_path}/{basename}", img)
        #         # cv2.imshow("img", img)
        #         cv2.waitKey(2)
        #     traj_datas["traj_euvs"] = traj_euvs
        #     write_json(traj_datas, traj_json_fn)

        # pts_gd_veh = self.iuvs_c2v(uvs_gd, self.rot_mtx_v2c, self.txyz_v2c)
        # assert np.sum(np.abs(pts_gd_veh - self.pts_gd_veh)) < 1.e-6, "There is some errors in 'iuvs_c2v'"

        cv2.imshow("img", img)
        cv2.imshow("retinal", ego_retinal_map)
        cv2.waitKey()
        pass


if __name__ == '__main__':
    cfgs = [cam_param_configs_sim, cam_param_ego_paper][0]

    cam_params, basenames_order = get_cam_params(cfgs)
    egr = EgoRetinalMap(cam_params, basenames_order)
    egr.run_single(img_idx=0)

    egr.run_single(img_idx=4)


