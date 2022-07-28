"""
PaperDataset link: http://humbi-dataset.s3.amazonaws.com/fut_loc.zip
"""

import os
import cv2
import json
import numpy as np


class DataConvert:
    def __init__(self, case_path, plot):
        self.vTR = self.read_traj(f"{case_path}/traj_prediction.txt")
        self.basenames = self.read_img_list(f"{case_path}/im_list.list")
        self.cam_params = self.read_cam_params( f"{case_path}/calib_fisheye.txt")

        self.img_fns = [f"{case_path}/im/{name}" for name in self.basenames]
        self.disp_fns = [f"{case_path}/disparity/{name}.disp.txt" for name in self.basenames]

        self.case_path = case_path
        self.size = len(self.vTR["vTr"])
        self.traj_json_path = f"{case_path}/traj_jsons"

        if not os.path.exists(self.traj_json_path):
            os.makedirs(self.traj_json_path)

        self.extract_trajs(plot)

    def read_disparity(self, filename):
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

    def read_img_list(self, filename):
        assert os.path.exists(filename), "File don't exits!"
        with open(filename, "r") as fs:
            vFilename = [line.replace("\n", "") for line in fs.readlines()]

        return vFilename

    def read_cam_params(self, filename):
        assert os.path.exists(filename)

        with open(filename, "r") as fs:
            lines = [line[:-1].split(": ") for line in fs.readlines()]

        cam_params = {"img_w": int(lines[0][1]), "img_h": int(lines[1][1])}
        for k, v in zip(["fx", "fy", "cx", "cy", "omega", "cx1", "cy2"], lines[2:]):
            cam_params[k] = float(v)

        cam_params["K"] = np.array([[cam_params["fx"], 0, cam_params["cx"]],
                                    [0, cam_params["fy"], cam_params["cy"]],
                                    [0, 0, 1]], dtype=np.float64)
        cam_params["R_rect"] = np.array([[0.9989, 0.0040, 0.0466],
                                         [-0.0040, 1.0000, -0.0002],
                                         [-0.0466, 0, 0.9989]], dtype=np.float64)

        cam_params["K*R_rect"] = np.dot(cam_params["K"], cam_params["R_rect"])
        return cam_params

    def extract_trajs(self, plot):
        for iFrame in range(49, self.size):
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
            trag_json = {"traj_uvs": trag_img}
            with open(f"{self.traj_json_path}/{basename}.json", "w") as fs:
                json.dump(trag_json, fs)

            if plot:
                cv2.polylines(img, [tr_gd.astype(np.int)], isClosed=False, color=(0, 0, 255), thickness=2)
                cv2.imshow("img", img)
                cv2.waitKey(2)
            pass


def convert_all_cases():
    root = "/home/xl/Disk/xl/fut_loc"
    cases = sorted([d for d in os.listdir(root)  if os.path.isdir(f"{root}/{d}")])
    for k, case in enumerate(cases):
        DataConvert(case_path=f"{root}/{case}", plot=True)
        print(f"k: {k}, case: {case} done!")

if __name__ == '__main__':
    convert_all_cases()
