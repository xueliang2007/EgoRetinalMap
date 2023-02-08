import os
import cv2 as cv
import glob
import json
import shutil
import numpy as np
import open3d as o3d


def load_json(json_fn):
    if not os.path.exists(json_fn):
        print(json_fn)
        raise ("json_file don't exits")

    with open(json_fn, "r") as f:
        json_datas = json.load(f)
    return json_datas


def write_json(data_dict, save_fn):
    with open(save_fn, 'w', encoding='utf-8') as fs:
        json.dump(data_dict, fs)
    return


def crop_pcd(pcd, x_min=-10, x_max=10, y_min=-0.5, y_max=3., z_min=-20, z_max=30, uniform_color=True, vis=True):
    if isinstance(pcd, str):
        pcd = o3d.io.read_point_cloud(pcd)
    bounding_ploy = np.array([
        [x_min, 0, z_min],
        [x_min, 0, z_max],
        [x_max, 0, z_max],
        [x_max, 0, z_min],
    ], dtype=np.float64).reshape([-1, 3])
    bounding_polygon = np.array(bounding_ploy, dtype=np.float64)

    vol = o3d.visualization.SelectionPolygonVolume()
    vol.axis_max = y_max
    vol.axis_min = y_min
    vol.orthogonal_axis = "Y"
    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
    # vol = o3d.visualization.read_selection_polygon_volume()
    """Json Example
        {
        "axis_max": 3.,
        "axis_min": -0.,
        "bounding_polygon":
            [
                [2.6509309513852526, 0.0, 1.6834473132326844],
                [2.5786428246917148, 0.0, 1.6892074266735244],
                [2.4625790337552154, 0.0, 1.6665777078297999]
            ],
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "Y",
        "version_major": 1,
        "version_minor": 0
    }
    """

    pcd_croped = vol.crop_point_cloud(pcd)

    if vis:
        if uniform_color:
            pcd.paint_uniform_color([1, 0, 0])
            pcd_croped.paint_uniform_color([0, 1, 0])
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
        o3d.visualization.draw_geometries([pcd, pcd_croped, mesh])
    return pcd_croped


def gather_results(root, dst_path):
    cases = sorted([d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}")])
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    size_dict = {}
    for k, case in enumerate(cases):
        folder = f"{dst_path}/{case}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for name in ["images", "traj_jsons_ego"]:
            if os.path.exists(f"{folder}/{name}"):
                shutil.rmtree(f"{folder}/{name}")

        shutil.copytree(f"{root}/{case}/data_for_gy", f"{folder}/images")
        shutil.copytree(f"{root}/{case}/traj_jsons_ego", f"{folder}/traj_jsons_ego")

        png_fns = glob.glob(f"{folder}/images/seq_*")
        json_fns = glob.glob(f"{folder}/traj_jsons_ego/*.json")
        num_png = len(png_fns)
        num_json = len(json_fns)
        assert num_png == 4 * num_json
        size_dict[case] = num_json

        print(f"{k}/{len(cases)} done!")

    print(size_dict)
    print(f"Total size: {sum(size_dict.values())}")
    print("Done!")


if __name__ == '__main__':
    root = "/home/xl/Disk/xl/all_case_results"
    dst_path = "/home/xl/Disk/xl/all_case_results"
    gather_results(root, dst_path)
