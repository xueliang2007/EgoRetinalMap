import os
import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform.rotation import Rotation as R_sci

from core.ego_pose import EgoPose


H, W = 480, 640
cam_k = np.array([[380.8157653808594, 0., 319.883148], [0., 380.8157653808594, 246.589081], [0., 0., 1.]])


# class dd

def extract_flowlidar_pose(pose_file):
    # image_name, qw, qx, qy, qz, tx, ty, tz
    if isinstance(pose_file, str):
        with open(pose_file, 'r') as f:
            lines = f.readlines()[1:]
    elif isinstance(pose_file, list):
        lines = pose_file

    n = len(lines)
    if n < 10:
        print("less pose record in this txt_log: %s\n" % os.path.basename(pose_file))
        return None

    ts_list, fm_list, pts_list = [None] * n, [None] * n, [None] * n
    for i in range(n):
        line = lines[i][:-1]
        while line.find("  ") != -1:
            line.replace("  ", " ")

        info = line.split(' ')
        quat_xyzw = [float(info[2]), float(info[3]), float(info[4]), float(info[1])]
        t_xyz = [float(info[5]), float(info[6]), float(info[7])]
        pxyz = -np.dot(R_sci.from_quat(quat_xyzw).as_matrix().T, t_xyz)

        frame_name = info[0]
        tsmp_cnt = frame_name.find('_')
        if tsmp_cnt != -1:
            time_stamp = float(frame_name[:tsmp_cnt])
        else:
            time_stamp = float(frame_name)

        pts_list[i] = pxyz
        ts_list[i] = time_stamp
        fm_list[i] = frame_name
    return ts_list, fm_list, pts_list


def plot_3d(xyzs_arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # x = np.arange(0, 200)
    # y = np.arange(0, 100)
    # x, y = np.meshgrid(x, y)
    # z = np.random.randint(0, 200, size=(100, 200))
    # y3 = np.arctan2(x, y)
    # ax.scatter(x, y, z, c=y3, marker='.', s=50, label='')
    for i, xyzs in enumerate(xyzs_arr):
        ax.scatter(xyzs[0], xyzs[1], xyzs[2], c=f"C{i}", marker='.', s=50, label='')
    plt.show()
    plt.close()


def get_ground_abcd_from_img_depth(img_path, depth_path, vis):
    img = cv.imread(img_path)
    depth = np.load(depth_path)
    depth *= 0.001  # mm -> m
    Zs = depth.reshape((-1, 1))

    mesh_xy = np.meshgrid(range(W), range(H))
    uvs_mtx = np.stack((mesh_xy[0], mesh_xy[1]), axis=2)
    uvs = uvs_mtx.reshape((-1, 2))
    xyzs = np.dot(np.linalg.inv(cam_k), np.insert(uvs, 2, 1, 1).T).T * Zs  # Nx3
    rgbs = np.array([img[v, u][::-1] for u, v in uvs]).astype(np.float) / 255
    del uvs_mtx, mesh_xy, img, depth, Zs

    mask_x = np.bitwise_and(xyzs[:, 0] > -4., xyzs[:, 0] < 4.)
    mask_y = xyzs[:, 1] > -2.
    mask_z = xyzs[:, 2] < 5.
    mask = np.bitwise_and(mask_x, np.bitwise_and(mask_y, mask_z))
    rgbs[np.bitwise_not(mask)] = [1, 0, 0]

    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyzs))
    pcd.colors = o3d.utility.Vector3dVector(rgbs)

    sample_idxs = np.arange(len(xyzs))[mask]
    pcd_crop = pcd.select_by_index(sample_idxs)
    del pcd, mask, mask_x, mask_y, mask_z

    num_bef = len(sample_idxs)
    pcd_ds = pcd_crop.voxel_down_sample(voxel_size=0.06)
    num_aft = len(pcd_ds.points)
    print(f"num of downsample, bef: {num_bef}, aft: {num_aft}")
    del pcd_crop

    gd_abcd, gd_idxs = pcd_ds.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)
    print("ground plane:", gd_abcd)

    if vis:
        pcd_gd = pcd_ds.select_by_index(gd_idxs, invert=False)
        pcd_ungd = pcd_ds.select_by_index(gd_idxs, invert=True)
        pcd_ungd.paint_uniform_color([1, 0, 0])
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([pcd_gd, pcd_ungd, mesh])
    a, b, c, d = gd_abcd
    cam_h = np.fabs(d) / np.sqrt(a * a + b * b + c * c)
    return gd_abcd, cam_h


def parse_flow_lidar_traj(traj_txt):
    """
    txt format:
        # basename_n, q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2], t_xyz[0], t_xyz[1], t_xyz[2],
        # vz_star, acceleration_z, cam_pose[0], cam_pose[1], cam_pose[2],
        # ground_plane[0], ground_plane[1], ground_plane[2],
        # get_ground_plane_average()[0], get_ground_plane_average()[1], get_ground_plane_average()[2],
        # rot_mat_euler[0], rot_mat_euler[1], rot_mat_euler[2], mean_track_distance, ground_pt_size, basename_p
    :param traj_txt:
    :return:
    """
    with open(traj_txt, "r") as fs:
        lines = fs.readlines()

    basenames, cam_hs = [], []
    gd_abcds, txyzs, R_mtxs = [], [], []
    for k, line in enumerate(lines):
        words = line.split(" ")

        basename = words[0]
        qxyzw = [float(words[i]) for i in [2, 3, 4, 1]]
        txyz = np.array([float(words[i]) for i in range(5, 8)]).reshape(3, 1)
        gd_abcd = np.array([float(words[13]), float(words[13]), float(words[13]), -1])  # Ax + By + Cz - 1 = 0
        cam_h = np.fabs(gd_abcd[3]) / np.linalg.norm(gd_abcd[:3])

        txyzs.append(txyz)
        cam_hs.append(cam_h)
        gd_abcds.append(gd_abcd)
        basenames.append(basename)
        R_mtxs.append(R_sci.from_quat(qxyzw).as_matrix())

    num = len(cam_hs)
    txyzs = np.array(txyzs).reshape((-1, 3))
    return basenames, R_mtxs, txyzs, cam_hs


def reproject_traj_t0(color_path, basenames, R_mtxs, txyzs, cam_hs):
    step = 20
    num = len(R_mtxs)
    movement_scale = 1.
    for k in range(0, num, step):
        if k != 0:
            break
        s = slice(k, k+step, 1)
        txyz_k = txyzs[s, :]
        R_mtx_k = R_mtxs[s]

        plot_3d(txyz_k)

        pts = []
        for i in range(1, step):
            p0toi = txyz_k[i]
            Rito0 = R_mtx_k[i].T
            pito0 = -np.dot(Rito0, p0toi)
            pts.append(pito0)
        pts = np.array(pts).reshape((-1, 3)) * movement_scale

        # R_v2c = R_sci.from_euler("X", 30, degrees=True).as_matrix()
        uvs = cv.projectPoints(pts, np.zeros(3), np.zeros(3), cam_k, np.zeros(5))[0].reshape((-1, 2))

        bsname_curr = basenames[s][0]
        img = cv.imread(f"{img_path}/{bsname_curr}")
        for i, uv in enumerate(uvs):
            uv_int = tuple(uv.astype(np.int).tolist())
            cv.circle(img, uv_int, 2, (0, 255, 0), -1)
            cv.putText(img, str(i), uv_int, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        cv.polylines(img, [uvs.astype(np.int32)], isClosed=False, color=(0, 0, 255), thickness=1)
        cv.imshow("traj", img)
        cv.waitKey()
        plot_3d(pts)
        pass


def reproject_traj(color_path, basenames, R_mtxs, txyzs, cam_hs):
    step = 20
    num = len(R_mtxs)
    movement_scale = 20.
    for k in range(0, num, step):
        if k + step >= num:
            continue
        s = slice(k, k+step, 1)
        txyz_k = txyzs[s, :]
        R_mtx_k = R_mtxs[s]

        pts = []
        t_xyzs_rel = [np.zeros(3)]
        R_mtxs_rel = [np.identity(3)]
        for i in range(1, step):
            delta_rot = np.dot(R_mtx_k[i], R_mtx_k[0].T)
            delta_txyz = txyz_k[i] - np.dot(delta_rot, txyz_k[0])
            pt = -np.dot(delta_rot.T, delta_txyz)
            pts.append(pt)
            R_mtxs_rel.append(delta_rot)
            t_xyzs_rel.append(delta_txyz)
        pts = np.array(pts).reshape((-1, 3)) * movement_scale

        # R_v2c = R_sci.from_euler("X", 30, degrees=True).as_matrix()
        uvs = cv.projectPoints(pts, np.zeros(3), np.zeros(3), cam_k, np.zeros(5))[0].reshape((-1, 2))

        bsname_curr = basenames[s][0]
        img = cv.imread(f"{img_path}/{bsname_curr}")
        for i, uv in enumerate(uvs):
            uv_int = tuple(uv.astype(np.int).tolist())
            cv.circle(img, uv_int, 2, (0, 255, 0), -1)
            cv.putText(img, str(i), uv_int, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        cv.polylines(img, [uvs.astype(np.int32)], isClosed=False, color=(0, 0, 255), thickness=1)
        cv.imshow("traj", img)
        cv.waitKey()
        pass


if __name__ == '__main__':
    # basename = "1656497951.001_000000.png"
    basename = "1656497956.205_000039.png"
    depth_fn = f"/Users/xl/Desktop/realsense-480/depths_0_sampled/{basename}"
    color_path = f"/Users/xl/Desktop/realsense-480/colors_0_sampled/{basename}"
    # plane_model, cam_h, disp = get_ground_from_pcd(rgb_img_fn, depth_fn, seg_mask=0, vis=1)

    img_path = "/Users/xl/Desktop/ShaoJun/2022-09-15-15-53-03/color-sampled"
    pose_txt = "/Users/xl/Desktop/ShaoJun/2022-09-15-15-53-03/flowlidar/pose_record_flowlidar.txt"
    # bsnames, R_mtxs, t_vecs, cam_hs = parse_flow_lidar_traj(pose_txt)
    # reproject_traj_t0(color_path, bsnames, R_mtxs, t_vecs, cam_hs)

    zoom_scale = 3.
    pose_infos = extract_flowlidar_pose(pose_txt)  # [time_stamp, frame_name, rpxyzw, pxyz]
    t_xyz = pose_infos[2]
    pts_v = np.array(t_xyz).reshape((-1, 3)) * zoom_scale
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_v))
    # o3d.visualization.draw_geometries([pcd, mesh])

    img_path = f"/Users/xl/Desktop/ShaoJun/2022-09-15-15-53-03/color_sampled/1663067563.996764_00315.jpg"
    depth_path = f"/Users/xl/Desktop/ShaoJun/2022-09-15-15-53-03/depth_sampled/1663067563.996764_00315.npy"
    gd_abcd = get_ground_abcd_from_img_depth(img_path, depth_path, vis=False)
    a, b, c, d = gd_abcd
    cam_h = np.fabs(d) / np.sqrt(a * a + b * b + c * c)

    epose = EgoPose(cam_k, cam_D=np.zeros(4), img_h=H, img_w=W)
    eulers_v2c, txyz_v2c = epose.get_ego_pose_v2c_without_instan_vel(gd_abcd)

    R_v2c = R_sci.from_euler("XYZ", eulers_v2c, degrees=True).as_matrix()
    pts_c = np.dot(R_v2c, pts_v.T).T + txyz_v2c
    pts_c_norm = pts_c / pts_c[:, 2].reshape((-1, 1))
    uvs = np.dot(cam_k, pts_c_norm.T).T[:, :2]

    img = cv.imread(img_path)
    for i, uv in enumerate(uvs):
        uv_int = tuple(uv.astype(np.int).tolist())
        cv.circle(img, uv_int, 2, (0, 255, 0), -1)
        cv.putText(img, str(i), uv_int, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv.polylines(img, [uvs.astype(np.int32)], isClosed=False, color=(0, 0, 255), thickness=1)
    cv.imshow("traj", img)
    cv.waitKey()
    pass






