import os
import cv2
import numpy as np
import open3d as o3d
#from open3d.geometry import create_rgbd_image_from_color_and_depth


def img_to_pointcloud(color_fn, depth_fn):

    img = o3d.io.read_image(color_fn)
    depth = o3d.io.read_image(depth_fn)
    rgb = o3d.geometry.Image(img)
    depth = o3d.geometry.Image(depth)

    zs = np.array(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0/25.5, depth_trunc=50.0, convert_rgb_to_intensity=False)

    cx, cy = 256., 256.
    fx, fy = 256., 256.
    img_w, img_h = 512, 512
    intrinsic = o3d.camera.PinholeCameraIntrinsic(img_w, img_h, fx, fy, cx, cy)

    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    o3d.visualization.draw_geometries([pc])
    print("afater draw geometry")


root = "../data"
color_fn = os.path.join(root, "rgb/1.png")
depth_fn = os.path.join(root, "depth/1.png")

img_to_pointcloud(color_fn, depth_fn)
rgb = cv2.imread(color_fn)
cv2.imshow("rgb", rgb)
cv2.waitKey(0)
