import os
import glob
import shutil
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
    "path_color": "rgb",
    "path_depth": "depth",
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
    "case": "20150401_walk_01",
    "path_color": "im",
    "seg_path": "seg",
    "path_depth": "disparity",
    "result_path": "results",
    "path_traj_img": "traj_jsons_img",
    "path_traj_ego": "traj_jsons_ego",
    "path_gy": "data_for_gy",
    "path_vis": "traj_vis",
}

cam_param_self_sj = {
    "img_h": 480,
    "img_w": 640,
    "D": np.zeros(5, dtype=np.float64),
    "fx": 380.815765,
    "fy": 380.815765,
    "cx": 319.883148,
    "cy": 246.589081,

    "rpy_v2c": [20., 0., 180.],  # deg

    "rho_max": 30.,
    "rho_min": 0.5,

    "prop": "slef",
    "img_suffix": "jpg",
    "ground_method": "pcd",

    "root": "/Users/xl/Desktop/ShaoJun",
    "case": "2022-09-15-15-53-03",
    "fl_pose_txt": "flowlidar/pose_record_flowlidar.txt",
    "path_color": "color_sampled",
    "path_depth": "depth_sampled",

    "path_seg": "",
    "path_vis": "traj_vis",
    "path_gy": "data_for_gy",
    "path_result": "results",
    "path_traj_img": "traj_jsons_img",
    "path_traj_ego": "traj_jsons_ego",
}


def get_cam_params(cam_cfgs):
    for folder in ["path_depth", "path_color", "seg_path", "path_result", "path_traj_img", "path_traj_ego", "path_gy", "path_vis"]:
        if cam_cfgs.get(folder) is not None:
            cam_cfgs[folder] = "%s/%s/%s" % (cam_cfgs["root"], cam_cfgs["case"], os.path.basename(cam_cfgs[folder]))

    for folder in ["path_vis", "path_gy", "path_traj_ego"]:
        # if os.path.exists(cam_cfgs[folder]):
        #     shutil.rmtree(cam_cfgs[folder])
        if not os.path.exists(cam_cfgs[folder]):
            os.makedirs(cam_cfgs[folder])

    if cam_cfgs["prop"] == "paper":
        calib_fisheye_txt = f"{cam_cfgs['path_color']}/../calib_fisheye.txt"
        cam_params = read_cam_params(calib_fisheye_txt)
        cam_cfgs.update(cam_params)
    elif cam_cfgs["prop"] == "sim":
        pass
    elif cam_cfgs["prop"] == "slef":
        cam_cfgs["fl_pose_txt"] = f"{cam_cfgs['root']}/{cam_cfgs['case']}/{cam_cfgs['fl_pose_txt']}"
        # cam_cfgs["fl_pose_txt"] = f"{cam_cfgs['root']}/{cam_cfgs['case']}/{cam_cfgs['fl_pose_txt']}"
        # cam_cfgs["fl_pose_txt"] = f"{cam_cfgs['root']}/{cam_cfgs['case']}/{cam_cfgs['fl_pose_txt']}"
    else:
        pass

    if cam_cfgs.get("K", None) is None:
        cam_cfgs["K"] = np.array([[cam_cfgs["fx"], 0, cam_cfgs["cx"]],
                                  [0, cam_cfgs["fy"], cam_cfgs["cx"]],
                                  [0, 0, 1]], dtype=np.float64)

    img_fns = glob.glob("%s/*.%s" % (cam_cfgs["path_color"], cam_cfgs["img_suffix"]))
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