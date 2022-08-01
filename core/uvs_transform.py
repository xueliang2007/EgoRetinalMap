import os
import cv2
import numpy as np
from configs import cam_param_configs_sim

cfgs = cam_param_configs_sim


class CoordTrans:
    def __init__(self, cam_k):
        self.cam_k = cam_k

    def uv_img2ego(self, remap_xy, uv):
        if len(uv.shape) != 2 or uv.shape[0] != 2:
            uv = uv.reshape(2, 1)

        iu0, iv0 = int(uv[0, 0]), int(uv[1, 0])
        iu1, iv1 = iu0 + 1, iv0 + 1
        uvs_img = np.array([[iu0, iv0], [iu0, iv1], [iu1, iv1], [iu1, iv0]], dtype=np.float32)

        eu0, ev0 = remap_xy[iu0, iv0, 0], remap_xy[iu0, iv0, 1]
        eu1, ev1 = remap_xy[iu1, iv1, 0], remap_xy[iu1, iv1, 1]
        uvs_ego = np.array([[eu0, ev0], [eu0, ev1], [eu1, ev1], [eu1, ev0]], dtype=np.float32)

        m = cv2.getPerspectiveTransform(uvs_img, uvs_ego)
        img_uv1 = np.insert(uv, 2, 1, axis=0)
        ego_uvx = np.dot(m, img_uv1)
        ego_uv1 = ego_uvx / ego_uvx[2, 0]
        ego_uv = ego_uv1[:2, 0]
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

    def uvs_img2ego(self, remap_xy, iuvs):
        euvs = [self.uv_img2ego(remap_xy, uv) for uv in iuvs]
        return euvs

    def uvs_ego2img(self, remap_xy, euvs):
        iuvs = [self.uv_ego2img(remap_xy, uv) for uv in euvs]
        return iuvs


def demo():
    ct = CoordTrans(cfgs["K"])
    uv_img = np.array([260.5, 450.5])
    remap_xy = np.load("../data/remap_xy.npy")

    uv_ego = ct.uv_img2ego(remap_xy, uv_img)
    uv_img2 = ct.uv_ego2img(remap_xy, uv_ego)


# if __name__ == '__main__':
#     demo()

