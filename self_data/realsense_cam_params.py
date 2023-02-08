import os
import glob
import shutil
import numpy as np

CamParamsTotal = {
    "baseline": -50.0648,
    "intrinsic_left.x.x": 0.497071,
    "intrinsic_left.x.y": 0.794316,
    "intrinsic_left.x.z": 0.501616,
    "intrinsic_left.y.x": 0.487021,
    "intrinsic_left.y.y": -0.0580017,
    "intrinsic_left.y.z": 0.0679893,
    "intrinsic_left.z.x": -0.000729303,
    "intrinsic_left.z.y": 0.00167767,
    "intrinsic_left.z.z": -0.021691,
    "intrinsic_right.x.x": 0.499367,
    "intrinsic_right.x.y": 0.797897,
    "intrinsic_right.x.z": 0.506448,
    "intrinsic_right.y.x": 0.493027,
    "intrinsic_right.y.y": -0.0576264,
    "intrinsic_right.y.z": 0.0666865,
    "intrinsic_right.z.x": -0.000395015,
    "intrinsic_right.z.y": 0.00167579,
    "intrinsic_right.z.z": -0.0213581,
    "rectified.0.fx": 957.204,
    "rectified.0.fy": 957.204,
    "rectified.0.height": 1080,
    "rectified.0.ppx": 971.836,
    "rectified.0.ppy": 527.868,
    "rectified.0.width": 1920,
    "rectified.1.fx": 638.136,
    "rectified.1.fy": 638.136,
    "rectified.1.height": 720,
    "rectified.1.ppx": 647.891,
    "rectified.1.ppy": 351.912,
    "rectified.1.width": 1280,
    "rectified.10.fx": 574.322,
    "rectified.10.fy": 574.322,
    "rectified.10.height": 0,
    "rectified.10.ppx": 367.102,
    "rectified.10.ppy": 352.721,
    "rectified.10.width": 0,
    "rectified.11.fx": 459.458,
    "rectified.11.fy": 459.458,
    "rectified.11.height": 0,
    "rectified.11.ppx": 293.681,
    "rectified.11.ppy": 282.177,
    "rectified.11.width": 0,
    "rectified.12.fx": 0,
    "rectified.12.fy": 0,
    "rectified.12.height": 400,
    "rectified.12.ppx": 0,
    "rectified.12.ppy": 0,
    "rectified.12.width": 640,
    "rectified.13.fx": 0,
    "rectified.13.fy": 0,
    "rectified.13.height": 576,
    "rectified.13.ppx": 0,
    "rectified.13.ppy": 0,
    "rectified.13.width": 576,
    "rectified.14.fx": 0,
    "rectified.14.fy": 0,
    "rectified.14.height": 720,
    "rectified.14.ppx": 0,
    "rectified.14.ppy": 0,
    "rectified.14.width": 720,
    "rectified.15.fx": 0,
    "rectified.15.fy": 0,
    "rectified.15.height": 1152,
    "rectified.15.ppx": 0,
    "rectified.15.ppy": 0,
    "rectified.15.width": 1152,
    "rectified.2.fx": 382.882,
    "rectified.2.fy": 382.882,
    "rectified.2.height": 480,
    "rectified.2.ppx": 324.735,
    "rectified.2.ppy": 235.147,
    "rectified.2.width": 640,
    "rectified.3.fx": 422.765,
    "rectified.3.fy": 422.765,
    "rectified.3.height": 480,
    "rectified.3.ppx": 429.228,
    "rectified.3.ppy": 234.642,
    "rectified.3.width": 848,
    "rectified.4.fx": 319.068,
    "rectified.4.fy": 319.068,
    "rectified.4.height": 360,
    "rectified.4.ppx": 323.945,
    "rectified.4.ppy": 175.956,
    "rectified.4.width": 640,
    "rectified.5.fx": 211.383,
    "rectified.5.fy": 211.383,
    "rectified.5.height": 240,
    "rectified.5.ppx": 214.614,
    "rectified.5.ppy": 117.321,
    "rectified.5.width": 424,
    "rectified.6.fx": 191.441,
    "rectified.6.fy": 191.441,
    "rectified.6.height": 240,
    "rectified.6.ppx": 162.367,
    "rectified.6.ppy": 117.574,
    "rectified.6.width": 320,
    "rectified.7.fx": 239.301,
    "rectified.7.fy": 239.301,
    "rectified.7.height": 270,
    "rectified.7.ppx": 242.959,
    "rectified.7.ppy": 131.967,
    "rectified.7.width": 480,
    "rectified.8.fx": 638.136,
    "rectified.8.fy": 638.136,
    "rectified.8.height": 800,
    "rectified.8.ppx": 647.891,
    "rectified.8.ppy": 391.912,
    "rectified.8.width": 1280,
    "rectified.9.fx": 478.602,
    "rectified.9.fy": 478.602,
    "rectified.9.height": 540,
    "rectified.9.ppx": 485.918,
    "rectified.9.ppy": 263.934,
    "rectified.9.width": 960,
    "world2left_rot.x.x": 0.999999,
    "world2left_rot.x.y": 0.00043766,
    "world2left_rot.x.z": 0.00133523,
    "world2left_rot.y.x": -0.000438896,
    "world2left_rot.y.y": 0.999999,
    "world2left_rot.y.z": 0.000925953,
    "world2left_rot.z.x": -0.00133482,
    "world2left_rot.z.y": -0.000926538,
    "world2left_rot.z.z": 0.999999,
    "world2right_rot.x.x": 0.999998,
    "world2right_rot.x.y": 1.18925e-05,
    "world2right_rot.x.z": -0.00174778,
    "world2right_rot.y.x": -1.35114e-05,
    "world2right_rot.y.y": 1,
    "world2right_rot.y.z": -0.000926234,
    "world2right_rot.z.x": 0.00174777,
    "world2right_rot.z.y": 0.000926257,
    "world2right_rot.z.z": 0.999998
}


class RS_Data:
    def __init__(self, root, color_folder_name, depth_folder_name, img_h, img_w, min_delta_ms):
        self.cam_params = self.get_param_with_curr_resolution(img_h, img_w)
        self.color_fns, self.depth_fns = self.sync_color_and_depth(root, color_folder_name, depth_folder_name,
                                                                   min_delta_ms)

    def get_param_with_curr_resolution(self, img_h, img_w):
        cam_id = 0
        while cam_id < 16:
            key_w = f"rectified.{cam_id}.width"
            key_h = f"rectified.{cam_id}.height"
            value_w = CamParamsTotal[key_w]
            value_h = CamParamsTotal[key_h]
            if value_h == img_h and value_w == img_w:
                break
            cam_id += 1
        assert cam_id < 16, "input img_hw error!"

        cam_params = {}
        for pi, key_j in zip(["fx", "fy", "ppx", "ppy", "width", "height"],
                             ["fx", "fy", "cx", "cy", "img_w", "img_h"]):
            key_i = f"rectified.{cam_id}.{pi}"
            cam_params[key_j] = CamParamsTotal[key_i]
        cam_params["cam_k"] = np.array([[cam_params["fx"], 0, cam_params["cx"]],
                                        [0, cam_params["fy"], cam_params["cy"]],
                                        [0, 0, 1]], dtype=np.float64)
        return cam_params

    def sync_color_and_depth(self, root, color_folder_name, depth_folder_name, min_delta_ms):
        color_fns = sorted(glob.glob(f"{root}/{color_folder_name}/*.png"))
        depth_fns = sorted(glob.glob(f"{root}/{depth_folder_name}/*.png"))

        sync_bsname_dict = {}
        color_num0 = len(color_fns)
        for color_fn in color_fns:
            color_bsname = os.path.basename(color_fn)
            ts_color = color_bsname.split("_")[-1].split(".")[0]
            sync_bsname_dict[ts_color] = [color_bsname]

        for depth_fn in depth_fns:
            depth_bsname = os.path.basename(depth_fn)
            ts_depth = depth_bsname.split("_")[-1].split(".")[0]
            if sync_bsname_dict.get(ts_depth) is None:
                continue
            else:
                sync_bsname_dict[ts_depth].append(depth_bsname)

        del color_fns, depth_fns

        timestamps = []
        sync_color_fns, sync_depth_fns = [], []
        for k, key in enumerate(sync_bsname_dict):
            value = sync_bsname_dict[key]
            if isinstance(value, list) and len(value) == 2:
                timestamps.append(float(key))
                sync_color_fns.append(os.path.abspath(f"{root}/{color_folder_name}/{value[0]}"))
                sync_depth_fns.append(os.path.abspath(f"{root}/{depth_folder_name}/{value[1]}"))

        del sync_bsname_dict

        if min_delta_ms <= 0:
            return sync_color_fns, sync_depth_fns

        sample_timestamps = [timestamps[0]]
        sampled_color_fns = [sync_color_fns[0]]
        sampled_depth_fns = [sync_depth_fns[0]]
        for k in range(1, len(timestamps)):
            if timestamps[k] - sample_timestamps[-1] < min_delta_ms:
                continue

            sample_timestamps.append(timestamps[k])
            sampled_color_fns.append(sync_color_fns[k])
            sampled_depth_fns.append(sync_depth_fns[k])

        color_num1 = len(sampled_color_fns)
        print(f"Sync and Sample, bef: {color_num0}, aft: {color_num1}")

        return sampled_color_fns, sampled_depth_fns

    def sample_to_new_path(self, color_path_new, depth_path_new):
        for path in [color_path_new, depth_path_new]:
            if not os.path.exists(path):
                os.makedirs(path)

        for k, [color_fn, depth_fn] in enumerate(zip(self.color_fns, self.depth_fns)):
            bsname_color = os.path.basename(color_fn)
            timestamp = bsname_color.split(".")[0].split("_")[-1]
            bsname_new = f"{timestamp[:-3]}.{timestamp[-3:]}_{k:06d}.png"

            dst_color_fn = f"{color_path_new}/{bsname_new}"
            dst_depth_fn = f"{depth_path_new}/{bsname_new}"
            shutil.copy(color_fn, dst_color_fn)
            shutil.copy(depth_fn, dst_depth_fn)

        print("Sample Done!")


if __name__ == '__main__':
    rs_data = RS_Data("/Users/xl/Desktop/realsense-480", "colors_0", "depths_0", 480, 848, 125)

    color_path_new = "/Users/xl/Desktop/realsense-480/colors_0_sampled"
    depth_path_new = "/Users/xl/Desktop/realsense-480/depths_0_sampled"
    # shutil.rmtree(color_path_new)
    # shutil.rmtree(depth_path_new)
    # rs_data.sample_to_new_path(color_path_new, depth_path_new)
    pass


