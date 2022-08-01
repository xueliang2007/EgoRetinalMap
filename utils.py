import os
import cv2
import json
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


def crop_pcd(pcd, x_min=-10, x_max=10, z_min=-20, z_max=30, uniform_color=True, vis=True):
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
    vol.orthogonal_axis = "Y"
    vol.axis_max = 3.
    vol.axis_min = -0.5
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
