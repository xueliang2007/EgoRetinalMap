import os
import glob
import numpy as np
from core.ego_data_convert import read_cam_params


cam_param_configs_sim = {
    "img_h": 512,
    "img_w": 512,
    "fx": 256.,
    "fy": 256.,
    "cx": 256.,
    "cy": 256.,
    "rho_max": 25.,
    "rho_min": 0.5,

    "prop": "sim",
    "D": np.zeros(5, dtype=np.float64),

    "depth_trunc": 50.,
    "depth_scalar": 1 / 25.5,
    "ground_method": "pcd",

    "root": "../data",
    "img_suffix": "png",
    "color_path": "rgb",
    "depth_path": "depth",
    "result_path": "results",
}

cam_param_ego_paper = {
    "img_h": 960,
    "img_w": 1280,
    "D": np.zeros(5, dtype=np.float64),

    "rpy_v2c": [180., 0., 0.],  # deg

    "rho_max": 60.,
    "rho_min": 2.5,

    "prop": "paper",
    "img_suffix": "png",
    "ground_method": "disp",

    # "root": "/Users/xl/Desktop/ShaoJun/ego_paper_data",
    "root": "/home/xl/Disk/xl/fut_loc",
    "case": "20150401_walk_00",
    "color_path": "im",
    "depth_path": "disparity",
    "result_path": "results",
    "path_traj": "traj_jsons",
    "path_gy": "data_for_gy",
    "path_vis": "traj_vis",
}


def get_cam_params(cam_cfgs):
    for folder in ["depth_path", "color_path", "result_path", "path_traj", "path_gy", "path_vis"]:
        if cam_cfgs.get(folder) is not None:
            cam_cfgs[folder] = "%s/%s/%s" % (cam_cfgs["root"], cam_cfgs["case"], os.path.basename(cam_cfgs[folder]))

    for folder in ["result_path", "path_traj", "path_gy", "path_vis", "path_gy"]:
        if not os.path.exists(cam_cfgs[folder]):
            os.makedirs(cam_cfgs[folder])

    if cam_cfgs["prop"] == "paper":
        calib_fisheye_txt = f"{cam_cfgs['color_path']}/../calib_fisheye.txt"
        cam_params = read_cam_params(calib_fisheye_txt)
        cam_cfgs.update(cam_params)
    elif cam_cfgs["prop"] == "sim":
        pass
    elif cam_cfgs["prop"] == "my":
        pass
    else:
        pass

    if cam_cfgs.get("K", None) is None:
        cam_cfgs["K"] = np.array([[cam_cfgs["fx"], 0, cam_cfgs["cx"]],
                                  [0, cam_cfgs["fy"], cam_cfgs["cx"]],
                                  [0, 0, 1]], dtype=np.float64)

    img_fns = glob.glob("%s/*.%s" % (cam_cfgs["color_path"], cam_cfgs["img_suffix"]))
    basenames = [[]] * len(img_fns)
    for i, img_fn in enumerate(img_fns):
        basename_i = os.path.basename(img_fn)
        key = basename_i
        if cam_cfgs["prop"] == "sim":
            key = "0" * (6 - len(os.path.splitext(basename_i)[0])) + basename_i
        basenames[i] = [key, basename_i]
    basenames = sorted(basenames, key=lambda x: x[0])
    basenames_order = [bsname[1] for bsname in basenames]

    return cam_cfgs, basenames_order