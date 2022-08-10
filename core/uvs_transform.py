import os
import cv2
import heapq
import numpy as np
from configs import cam_param_configs_sim, cam_param_ego_paper, get_cam_params


class CoordTrans:
    def __init__(self, cam_k, img_h, img_w):
        self.cam_k = cam_k
        self.img_h = img_h
        self.img_w = img_w

    def uv_img2ego(self, remap, uv):
        if isinstance(uv, list):
            uv = np.array(uv)
        if len(uv.shape) != 2 or uv.shape[0] != 2:
            uv = uv.reshape(2, 1)

        if not (0 < uv[0, 0] < self.img_w-1 and 0 < uv[1, 0] < self.img_h-1):
            return None

        iu0, iv0 = int(uv[0, 0]), int(uv[1, 0])
        iu1, iv1 = iu0 + 1, iv0 + 1

        umax, umin = np.max(remap[:, :, 0]), np.min(remap[:, :, 0])
        vmax, vmin = np.max(remap[:, :, 1]), np.min(remap[:, :, 1])
        if not (umin < uv[0, 0] < umax and vmin < uv[1, 0] < vmax):
            return None

        uvs_ego, uvs_img = [], []
        rmp_h, rmp_w = remap.shape[:2]
        # uvs_img = np.array([[iu0, iv0], [iu0, iv1], [iu1, iv1], [iu1, iv0]], dtype=np.float32)

        c0, r0 = divmod(np.argmin(np.abs(remap[:, :, 0] - uv[0, 0])), rmp_w)
        value0 = remap[c0, r0, 0]
        if value0 < uv[0, 0]:
            eu0, ev0 = c0 - 1, r0
            eu1, ev1 = c0, r0 + 1
        else:
            eu0, ev0 = c0, r0 - 1
            eu1, ev1 = c0 + 1, r0
        value00 = remap[eu0, ev0, 0]
        value10 = remap[eu0, ev1, 0]
        value11 = remap[eu1, ev1, 0]
        value01 = remap[eu1, ev0, 0]
        # uvs_img = np.array([[iu0, iv0], [iu0, iv1], [iu1, iv1], [iu1, iv0]], dtype=np.float32)
        # uvs_img = np.array([remap[vi, ui, 0] for ui, vi in [[eu0, ev0], [eu0, ev1], [eu1, ev1], [eu1, ev0]]], dtype=np.float32)
        uvs_img.append(np.array([remap[ui, vi] for ui, vi in [[eu0, ev0], [eu0, ev1], [eu1, ev1], [eu1, ev0]]], dtype=np.float32))
        # uvs_ego.append(np.array([[eu0, ev0], [eu0, ev1], [eu1, ev1], [eu1, ev0]], dtype=np.float32))
        uvs_ego.append(np.array([[ev0, eu0], [ev0, eu1], [ev1, eu1], [ev1, eu0]], dtype=np.float32))

        c1, r1 = divmod(np.argmin(np.abs(remap[:, :, 1] - uv[1, 0])), rmp_w)
        value1 = remap[c1, r1, 1]
        if value1 > uv[1, 0]:
            eu0, ev0 = c1 - 1, r1
            eu1, ev1 = c1, r1 + 1
        else:
            eu0, ev0 = c1, r1 - 1
            eu1, ev1 = c1 + 1, r1
        uvs_img.append(np.array([remap[ui, vi] for ui, vi in [[eu0, ev0], [eu0, ev1], [eu1, ev1], [eu1, ev0]]], dtype=np.float32))
        uvs_ego.append(np.array([[eu0, ev0], [eu0, ev1], [eu1, ev1], [eu1, ev0]], dtype=np.float32))

        ego_uv = []
        for i, [src_i, dst_i] in enumerate(zip(uvs_img, uvs_ego)):
            m = cv2.getPerspectiveTransform(src_i, dst_i)
            img_uv1 = np.insert(uv, 2, 1, axis=0)
            ego_uvx = np.dot(m, img_uv1)
            ego_uv1 = ego_uvx / ego_uvx[2, 0]
            ego_uv.append(ego_uv1[i, 0])
        return ego_uv

    def uv_ego2img(self, remap_xy, uv):
        if len(uv.shape) != 2 or uv.shape[0] != 2:
            uv = uv.reshape(2, 1)

        eu0, ev0 = int(uv[0, 0]), int(uv[1, 0])
        eu1, ev1 = eu0 + 1, ev0 + 1
        uvs_ego = np.array([[eu0, ev0], [eu0, ev1], [eu1, ev1], [eu1, ev0]], dtype=np.float32)

        iv0 = np.argmin(np.abs(remap_xy[-1, :, 0] - eu0))
        iu0 = np.argmin(np.abs(remap_xy[:, iv0, 1] - ev0))
        iv1 = iv0 + 1
        iu1 = iu0 + 1
        uvs_img = np.array([[iu0, iv0], [iu0, iv1], [iu1, iv1], [iu1, iv0]], dtype=np.float32)

        m = cv2.getPerspectiveTransform(uvs_ego, uvs_img)
        ego_uv1 = np.insert(uv, 2, 1, axis=0)
        img_uvx = np.dot(m, ego_uv1)
        img_uv1 = img_uvx / img_uvx[2, 0]
        img_uv = img_uv1[:2, 0]
        return img_uv

    def iuvs_c2v(self, obj_uvs, rot_mtx_v2c, txyz_v2c):
        """
        根据外参参数，通过目标点在图像上的投影恢复他在车辆坐标系下真实的3D坐标，其中假设3D坐标的Y为0
        """
        assert obj_uvs.shape[1] == 2  # Nx2

        K_inv = np.linalg.inv(self.cam_k)
        uv1s = np.insert(obj_uvs, 2, 1, axis=1)
        pts_prime = np.dot(K_inv, uv1s.T).T

        # solve: 's * (r01*x' + r11*y' + r21) = r01*tx + r11*ty + r21*tz' -> 's * left = right' -> s?
        right = np.dot(txyz_v2c, rot_mtx_v2c[:, 1])[0]
        lefts = np.dot(pts_prime, rot_mtx_v2c[:, 1])
        scales = right / lefts
        pts_cam = np.zeros_like(pts_prime)
        for i in range(3):
            pts_cam[:, i] = pts_prime[:, i] * scales
        pts_veh = np.dot(rot_mtx_v2c.T, (pts_cam - txyz_v2c).T).T

        return pts_veh

    def invert_remap(self, remap_xy: np.ndarray):
        I = np.zeros_like(remap_xy)
        I[:, :, 1], I[:, :, 0] = np.indices(remap_xy.shape[:2])
        remap_xy_inv = np.copy(I)
        for i in range(10):
            remap_xy_inv += I - cv2.remap(remap_xy, remap_xy_inv, None, interpolation=cv2.INTER_LINEAR)
        return remap_xy_inv

    def uvs_warp(self, remap_xy, uvs_ip, method):
        uvs_op = []
        if method == "e2i":
            for i, uv in enumerate(uvs_ip):
                r = self.uv_ego2img(remap_xy, uv)
                uvs_op.append(r)
        elif method == "i2e":
            xymap_inv = self.invert_remap(remap_xy)

            # unwarped = cv2.remap(warped, xymap_inv, None, cv2.INTER_LINEAR)
            for i, uv in enumerate(uvs_ip):
                r = self.uv_img2ego(remap_xy, uv)
                if r is not None:
                    uvs_op.append(r)
        else:
            assert False
        return uvs_op


def demo():
    cfgs, basenames = get_cam_params(cam_param_ego_paper)
    ct = CoordTrans(cfgs["K"], cfgs["img_h"], cfgs["img_w"])

    data = np.load("../data/remap_xy.npz")
    remap_xy = data["remap_xy"]
    traj_iuvs = data["traj_iuvs"]
    ct.uvs_warp(remap_xy, traj_iuvs, method="i2e")
    pass

    # uv_ego = ct.uv_img2ego(remap_xy, uv_img)
    # uv_img2 = ct.uv_ego2img(remap_xy, uv_ego)


if __name__ == '__main__':
    demo()

