import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def convert_xy(idx, center, step, res):
    return (idx - (res / 2)) * step + center

def convert_ij(xy, center, step, res):
    return (np.round((xy - center) / step) + res / 2).astype(np.int32)

def convert_binary_map(xy, center, step, res):
    binary_map = np.zeros((res, res))
    idx = convert_ij(xy, center, step, res)
    binary_map[idx[:, 0], idx[:, 1]] = 1

    return binary_map

def scatter_3d(ax, xyz):
    xdata = xyz[:, 0]
    ydata = xyz[:, 1]
    zdata = xyz[:, 2]
    ax.scatter3D(xdata, ydata, zdata, marker='s')

def plot_layout(xyz, planes, plot_pcd=False, down_sample_size=0.5):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for plane in planes:
        p = plane[0]

        if (abs(p[0]) > abs(p[1]) and abs(p[0]) > abs(p[2])):
            inliers = plane[1]
            y = np.linspace(np.amin(inliers[:, 1]), np.amax(inliers[:, 1]), 2)
            z = np.linspace(np.amin(inliers[:, 2]), np.amax(inliers[:, 2]), 2)
            Y, Z = np.meshgrid(y,z)
            X = (-p[3] - p[2]*Z - p[1]*Y) / p[0]
            ax.plot_surface(X, Y, Z)
        elif (abs(p[1]) > abs(p[0]) and abs(p[1]) > abs(p[2])):
            inliers = plane[1]
            x = np.linspace(np.amin(inliers[:, 0]), np.amax(inliers[:, 0]), 2)
            z = np.linspace(np.amin(inliers[:, 2]), np.amax(inliers[:, 2]), 2)
            X, Z = np.meshgrid(x,z)
            Y = (-p[3] - p[0]*X - p[2]*Z) / p[1]
            ax.plot_surface(X, Y, Z)
        else:
            inliers = plane[1]
            x = np.linspace(np.amin(inliers[:, 0]), np.amax(inliers[:, 0]), 2)
            y = np.linspace(np.amin(inliers[:, 1]), np.amax(inliers[:, 1]), 2)
            X, Y = np.meshgrid(x,y)
            Z = (-p[3] - p[0]*X - p[1]*Y) / p[2]
            ax.plot_surface(X, Y, Z)

    if plot_pcd:
        pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pcl = pcl.voxel_down_sample(voxel_size=down_sample_size)
        scatter_3d(ax, np.asarray(pcl.points))

    plt.show()

def plot_plane_points(planes):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    points = None
    for p in planes:
        if points is None:
            points = p[1].copy()
        else:
            points = np.concatenate((points, p[1]), axis=0)

    scatter_3d(ax, points)
    plt.show()

def visulize_pcd(pcl):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl))
    o3d.visualization.draw_geometries([pcd], width=1440, height=900)

def vpos2xyz(a, mat):
    # x = j, i, raw depth
    coords = (a[0] - 0.5, a[1] - 0.5)
    uv = (coords[1] * 2 * np.pi, -coords[0] * np.pi)
    depth = a[2] / 512

    x = np.cos(uv[1]) * np.sin(uv[0])
    y = np.sin(uv[1])
    z = np.cos(uv[1]) * np.cos(uv[0])
    xyz = np.dot(mat, np.array([x * depth, y * depth, -z * depth, 1.0]))
    # xyz = np.array([x * depth, y * depth, -z * depth])
    return xyz[:3]

def multi_view_pcd(gibson_mesh_path, model, room_to_pano, room_id):
    for pano in room_to_pano[room_id]:
        depth_path = os.path.join(gibson_mesh_path, model, 'pano', 'mist', f'point_{pano}_view_equirectangular_domain_mist.png')
        depth_pano = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        height, width = depth_pano.shape

        json_path = os.path.join(gibson_mesh_path, model, 'pano', 'points', f'point_{pano}.json')
        rt_matrix = np.array(pose_tools.load_json_pose(json_path)[1]['camera_rt_matrix'])
        inv_rt_matrix = np.linalg.inv(rt_matrix)
        # print(inv_rt_matrix)

        depth_pano = np.expand_dims(depth_pano, axis=2)
        pos_map = np.indices((height, width)).astype(np.float)
        pos_map[0] = pos_map[0] / height
        pos_map[1] = pos_map[1] / width
        pos_map = pos_map.transpose((1,2,0))
        pcd = np.concatenate((pos_map, depth_pano), axis=2)
        
        print(f'Generating {pano} pcd')
        pcl = [[vpos2xyz(p, rt_matrix) for p in row] for row in pcd]
        pcl = np.array(pcl)
        np.save(pano, pcl)
        print(f'Saved {pano} pcd')