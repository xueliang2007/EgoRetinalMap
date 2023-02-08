import os
import cv2 as cv
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

        umax, umin = np.max(remap[:, :, 0]), np.min(remap[:, :, 0])
        vmax, vmin = np.max(remap[:, :, 1]), np.min(remap[:, :, 1])
        if not (umin < uv[0, 0] < umax and vmin < uv[1, 0] < vmax):
            return None

        vertexs = [None] * 4  # tl, tr, br, bl
        update_cnt = [0] * 4
        errs = [0, 0, 0, 0, 10000]
        rmp_h, rmp_w = remap.shape[:2]
        remap_vec = remap.reshape((-1, 2))
        idxs = np.argpartition(np.sum(np.fabs(remap_vec - uv[:, 0]), axis=1), 250)[:250]
        for k, i in enumerate(idxs):
            uvi = remap_vec[i]
            if uvi[0] < uv[0, 0] and uvi[1] < uv[1, 0]:
                update_cnt[0] += 1
                diff_curr0 = uv[0, 0] + uv[1, 0] - uvi[0] - uvi[1]
                if vertexs[0] is None:
                    vertexs[0] = i
                    errs[0] = diff_curr0
                else:
                    uv_best = remap_vec[vertexs[0]]
                    diff_min0 = uv[0, 0] + uv[1, 0] - uv_best[0] - uv_best[1]
                    if diff_curr0 < diff_min0:
                        vertexs[0] = i
                        errs[0] = diff_curr0
            elif uvi[0] > uv[0, 0] and uvi[1] < uv[1, 0]:
                update_cnt[1] += 1
                diff_curr1 = uvi[0] - uv[0, 0] + uv[1, 0] - uvi[1]
                if vertexs[1] is None:
                    vertexs[1] = i
                    errs[1] = diff_curr1
                else:
                    uv_best = remap_vec[vertexs[1]]
                    diff_min1 = uv_best[0] - uv[0, 0] + uv[1, 0] - uv_best[1]
                    if diff_curr1 < diff_min1:
                        vertexs[1] = i
                        errs[1] = diff_curr1
            elif uvi[0] > uv[0, 0] and uvi[1] > uv[1, 0]:
                update_cnt[2] += 1
                diff_curr2 = uvi[0] - uv[0, 0] + uvi[1] - uv[1, 0]
                if vertexs[2] is None:
                    vertexs[2] = i
                    errs[2] = diff_curr2
                else:
                    uv_best = remap_vec[vertexs[2]]
                    diff_min2 = uv_best[0] - uv[0, 0] + uv_best[1] - uv[1, 0]
                    if diff_curr2 < diff_min2:
                        vertexs[2] = i
                        errs[2] = diff_curr2
            elif uvi[0] < uv[0, 0] and uvi[1] > uv[1, 0]:
                update_cnt[3] += 1
                diff_curr3 = uv[0, 0] - uvi[0] + uvi[1] - uv[1, 0]
                if vertexs[3] is None:
                    vertexs[3] = i
                    errs[3] = diff_curr3
                else:
                    uv_best = remap_vec[vertexs[3]]
                    diff_min3 = uv[0, 0] - uv_best[0] + uv_best[1] - uv[1, 0]
                    if diff_curr3 < diff_min3:
                        vertexs[3] = i
                        errs[3] = diff_curr3
            if all(vertexs):
                sum_curr = sum(errs[:-1])
                if sum_curr < errs[4]:
                    errs[4] = sum_curr
                    # print(f"k: {k:03d}, sum: {errs[4]:.3f}, {errs[0]:.2f}, {errs[1]:.2f}, {errs[2]:.2f}, {errs[3]:.2f}")
                if sum_curr < 10:
                    break

        if not all(vertexs):
            return None

        PRINT = 0
        if PRINT:
            print("update_cnt: ", ", ".join([str(c) for c in update_cnt]))
            print(uv.ravel())
            for ii in range(2):
                txt = f"{'xy'[ii]}: "
                for jj in range(4):
                    txt += f"{remap_vec[vertexs[jj], ii]:.4f}, "
                print(txt[:-2])

        uvs_img, uvs_ego = [], []
        for idx in vertexs:
            ri, ci = divmod(idx, rmp_w)
            uvs_ego.append([ri, ci])
        uvs_img = np.array([remap_vec[idx] for idx in vertexs], dtype=np.float32)
        uvs_ego = np.array(uvs_ego, dtype=np.float32)

        m = cv.getPerspectiveTransform(uvs_img, uvs_ego)
        img_uv1 = np.insert(uv, 2, 1, axis=0)
        ego_uvx = np.dot(m, img_uv1)
        ego_uv1 = ego_uvx / ego_uvx[2, 0]
        ego_uv = ego_uv1[:2, 0][::-1]
        return [ego_uv[0], ego_uv[1]]

    def uv_ego2img(self, remap_xy, uv):
        if len(uv.shape) != 2 or uv.shape[0] != 2:
            uv = uv.reshape(2, 1)

        eu0, ev0 = int(uv[0, 0]), int(uv[1, 0])
        eu1, ev1 = eu0 + 1, ev0 + 1
        uvs_ego = [[ev0, eu0], [ev1, eu0], [ev1, eu1], [ev0, eu1]]
        uvs_img = np.array([remap_xy[u, v] for u, v in uvs_ego], dtype=np.float32)
        uvs_ego = np.array(uvs_ego, dtype=np.float32)

        m = cv.getPerspectiveTransform(uvs_ego, uvs_img)
        ego_uv1 = np.insert(uv[::-1], 2, 1, axis=0)
        img_uvx = np.dot(m, ego_uv1)
        img_uv1 = img_uvx / img_uvx[2, 0]
        img_uv = [img_uv1[0, 0], img_uv1[1, 0]]
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

    def uvs_warp_verify(self):
        data = np.load("../data/seq_20150401_walk_01_000500_l.png.rect.png.npz")
        remap_xy, traj_iuvs = data["remap_xy"], data["traj_iuvs"]

        mask = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        cv.polylines(mask, [traj_iuvs.astype(np.int)], isClosed=False, color=(255, 255, 255), thickness=3)
        ego_mask = cv.remap(mask, remap_xy[:, :, 0], remap_xy[:, :, 1], cv.INTER_LINEAR,
                             borderMode=cv.BORDER_CONSTANT, borderValue=[0, 0, 0])
        uvs_op = []
        for i, uv in enumerate(traj_iuvs):
            r = self.uv_img2ego(remap_xy, uv)
            if r is None:
                r = [-1, -1]
            uvs_op.append(r)
        uvs_op = np.array(uvs_op)

        # Verify-1. Plot on egoMap to verfiy, PASS
        for u, v in uvs_op.astype(np.int):
            if u < 0 and v < 0:
                continue
            cv.circle(ego_mask, (u, v), 3, (0, 0, 255), -1)
        cv.imshow("ego_mask", ego_mask)
        cv.waitKey(10)

        # Verify-2. inverse transform, compare with itself, PASS: uvs_op2 should equal to uv_op_gt
        uv_op_gt = traj_iuvs[uvs_op[:, 0] > 0]
        uvs_op = uvs_op[uvs_op[:, 0] > 0]
        uvs_op2 = []
        for i, uv in enumerate(uvs_op):
            r = self.uv_ego2img(remap_xy, uv, uv_op_gt[i])
            uvs_op2.append(r)
        uvs_op2 = np.array(uvs_op2)
        err = np.abs(uvs_op2 - uv_op_gt)
        assert np.max(err) < 3., f"max_err: {np.max(err):.2f}"
        print("Verify Pass")

    def uvs_warp(self, remap_xy, uvs_ip, method):
        """
        unvalid in uvs_op is [-1, -1]
        """
        uvs_op = []
        if method == "e2i":
            for i, uv in enumerate(uvs_ip):
                r = self.uv_ego2img(remap_xy, uv)
                uvs_op.append(r)
        elif method == "i2e":
            for i, uv in enumerate(uvs_ip):
                r = self.uv_img2ego(remap_xy, uv)
                if r is None:
                    r = [-1, -1]
                uvs_op.append(r)
        else:
            assert False
        return uvs_op


def demo():
    cfgs, basenames = get_cam_params(cam_param_ego_paper)
    ct = CoordTrans(cfgs["K"], cfgs["img_h"], cfgs["img_w"])
    ct.uvs_warp_verify()
    pass

    # uv_ego = ct.uv_img2ego(remap_xy, uv_img)
    # uv_img2 = ct.uv_ego2img(remap_xy, uv_ego)


if __name__ == '__main__':
    demo()

