"""
PaperDataset link: http://humbi-dataset.s3.amazonaws.com/fut_loc.zip

Replace Matlab with Python
"""

import os
import cv2
import json
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


def read_cam_params(filename):
    assert os.path.exists(filename)

    with open(filename, "r") as fs:
        lines = [line[:-1].split(": ") for line in fs.readlines()]

    cam_params = {"img_w": int(lines[0][1]), "img_h": int(lines[1][1])}
    for k, v in zip(["fx", "fy", "cx", "cy", "omega", "cx1", "cy2"], lines[2:]):
        cam_params[k] = float(v[1])

    cam_params["K"] = np.array([[cam_params["fx"], 0, cam_params["cx"]],
                                [0, cam_params["fy"], cam_params["cy"]],
                                [0, 0, 1]], dtype=np.float64)
    cam_params["R_rect"] = np.array([[0.9989, 0.0040, 0.0466],
                                     [-0.0040, 1.0000, -0.0002],
                                     [-0.0466, 0, 0.9989]], dtype=np.float64)
    cam_params["K*R_rect"] = np.dot(cam_params["K"], cam_params["R_rect"])
    return cam_params


def get_ground_from_disp(fx, fy, cx, cy, img_w, img_h, bgr_fn, disp_fn, vis):
    basename = os.path.basename(bgr_fn)
    disp = DataConvert.read_disparity(disp_fn)
    disp = cv2.resize(disp, (img_w, img_h))

    Tx = -0.1  # m
    f = (fx + fy) * 0.5
    Q = np.array([[1.0, 0.0, 0.0, -cx],
                  [0.0, 1.0, 0.0, -cy],
                  [0.0, 0.0, 0.0,   f],
                  [0.0, 0.0, -1.0 / Tx, 0.0]])
    depth_map = cv2.reprojectImageTo3D(disp.astype(np.float32), Q, handleMissingValues=True)

    valid_roi = [100, 100, img_w-100, img_h-100]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.rectangle(mask, (valid_roi[0], valid_roi[1]), (valid_roi[2], valid_roi[3]), 1, -1)
    mask = mask.astype(np.bool)

    mask[depth_map[:, :, 2] < 0] = False
    mask[depth_map[:, :, 2] > 80] = False
    pts = depth_map[mask, ::].reshape((-1, 3))

    print(basename)
    print(f"before, num: {len(pts)}, mask: {np.sum(mask)}")
    for xyz_idx in range(3):
        print(f"\t{'XYZ'[xyz_idx]}, min: {np.min(pts[:, xyz_idx])}, max: {np.max(pts[:, xyz_idx])}")

    xyz_minmax = [[-15, 15], [-4, 5], [3, 25]]
    for xyz_idx in range(3):
        mask[depth_map[:, :, xyz_idx] < xyz_minmax[xyz_idx][0]] = False
        mask[depth_map[:, :, xyz_idx] > xyz_minmax[xyz_idx][1]] = False

    pts = depth_map[mask, ::].reshape((-1, 3))
    print(f"after, num: {len(pts)}")
    for xyz_idx in range(3):
        print(f"\t{'XYZ'[xyz_idx]}, min: {np.min(pts[:, xyz_idx])}, max: {np.max(pts[:, xyz_idx])}")

    del depth_map, disp

    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    pcd = pcd.voxel_down_sample(voxel_size=0.15)
    print(f"down, num: {len(pcd.points)}")

    plane_model, inlier_idxs = pcd.segment_plane(distance_threshold=0.06, ransac_n=3, num_iterations=500)
    a, b, c, d = plane_model
    cam_h = np.fabs(d) / np.sqrt(a * a + b * b + c * c)

    if vis:
        inlier_pcd = pcd.select_by_index(inlier_idxs)
        inlier_pcd.paint_uniform_color([1.0, 0, 0])
        outlier_pcd = pcd.select_by_index(inlier_idxs, invert=True)
        outlier_pcd.paint_uniform_color([0, 1., 0])
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
        # o3d.visualization.draw_geometries([pcd, mesh])
        o3d.visualization.draw_geometries([mesh, inlier_pcd, outlier_pcd])
    return plane_model, cam_h


class DataConvert:
    def __init__(self, case_path):
        self.vTR = self.read_traj(f"{case_path}/traj_prediction.txt")
        self.basenames = self.read_img_list(f"{case_path}/im_list.list")
        self.cam_params = read_cam_params(f"{case_path}/calib_fisheye.txt")

        self.img_fns = [f"{case_path}/im/{name}" for name in self.basenames]
        self.disp_fns = [f"{case_path}/disparity/{name}.disp.txt" for name in self.basenames]

        self.case_path = case_path
        self.size = len(self.vTR["vTr"])
        self.traj_vis_path = f"{case_path}/traj_vis"
        self.traj_json_path = f"{case_path}/traj_jsons"

        for folder_path in [self.traj_json_path, self.traj_vis_path]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def read_img_list(self, filename):
        assert os.path.exists(filename), "File don't exits!"
        with open(filename, "r") as fs:
            vFilename = [line.replace("\n", "") for line in fs.readlines()]

        return vFilename

    @staticmethod
    def read_disparity(filename):
        assert filename.endswith(".txt")
        assert os.path.exists(filename), "File don't exits!"

        with open(filename, "r") as fs:
            lines = fs.readlines()

        disp = []
        for line in lines:
            while not line[-1].isalnum():
                line = line[:-1]
            line = [float(w) for w in line.split(",")]
            disp.append(line)

        disp = np.array(disp).reshape((480, 640))
        return disp

    def read_traj(self, filename):
        assert filename.endswith(".txt")
        assert os.path.exists(filename), "File don't exits!"

        with open(filename, "r") as fs:
            lines = fs.readlines()

        n = int(lines[0].split(" ")[1])
        vTakenFrame = [None] * n
        vTr = [{} for _ in range(n)]

        for i, line in enumerate(lines[1:]):
            line = line.split(" ")
            iFrame = int(line[0])
            up = np.array([[float(w) for w in line[1:4]]])

            frame, XYZ, uv = [], [], []
            nTrjFrame = int(line[4])
            if nTrjFrame > 0:
                trajs = np.array([[float(line[nj]) for nj in range(5 + ni * 6, 5 + 6 * (ni + 1))] for ni in range(nTrjFrame)])
                frame = trajs[:, 0].astype(np.int).tolist()
                XYZ = trajs[:, 1:4]
                uv = trajs[:, 4:6]

            vTr[iFrame] = {
                "up": up,
                "uv": uv,
                "XYZ": XYZ,
                "frame": frame
            }
            vTakenFrame[iFrame] = iFrame

        vTR = {"vTr": vTr, "vTakenFrame": vTakenFrame}
        return vTR

    def extract_trajs(self, plot):
        for iFrame in range(self.size):
            tr = self.vTR["vTr"][iFrame]
            img_fn = self.img_fns[iFrame]
            if not os.path.exists(img_fn) or not len(tr["XYZ"]):
               continue

            img = cv2.imread(img_fn)
            tr_gd = tr["XYZ"] - tr["up"]
            tr_gd = np.dot(self.cam_params["K*R_rect"], tr_gd.T).T  # Nx3
            tr_gd = tr_gd[tr_gd[:, 2] > 0]
            tr_gd /= tr_gd[:, 2].reshape((-1, 1))
            tr_gd = tr_gd[:, :2]

            basename = self.basenames[iFrame]
            trag_img = [[u, v] for u, v in tr_gd]
            trag_json = {"traj_iuvs": trag_img}
            with open(f"{self.traj_json_path}/{basename}.json", "w") as fs:
                json.dump(trag_json, fs)

            if plot:
                cv2.polylines(img, [tr_gd.astype(np.int)], isClosed=False, color=(0, 0, 255), thickness=2)
                cv2.imwrite(f"{self.traj_vis_path}/{basename}", img)
                cv2.imshow("img", img)
                cv2.waitKey(2)
            pass


def convert_all_cases():
    # root = "/home/xl/Disk/xl/fut_loc"
    root = "/Users/xl/Desktop/ShaoJun/ego_paper_data"
    cases = sorted([d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}")])
    for k, case in enumerate(cases):
        dc = DataConvert(case_path=f"{root}/{case}")
        dc.extract_trajs(plot=False)

        print(f"k: {k}, case: {case} done!")


if __name__ == '__main__':
    convert_all_cases()
