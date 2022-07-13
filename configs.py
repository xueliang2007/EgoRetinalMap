import numpy as np


cam_param_configs_sim = {
    "img_h": 512,
    "img_w": 512,
    "fx": 256.,
    "fy": 256.,
    "cx": 256.,
    "cy": 256.,
    "D": np.zeros(5, dtype=np.float64),

    "rpy_v2c": [180., 20., 0.],  # deg

    "depth_trunc": 50.,
    "depth_scalar": 1 / 25.5,

    "root": "./data",
    "color_path": "rgb",
    "depth_path": "depth",
    "img_suffix": "png",

    "result_path": "results"
}

