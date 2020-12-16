import numpy as np
import pyransac3d as pyrsc

def ransac_find_planes(pcd):
    planes = []

    max_x = np.amax(pcd[:, 0])
    min_x = np.amin(pcd[:, 0])
    max_y = np.amax(pcd[:, 1])
    min_y = np.amin(pcd[:, 1])
    max_z = np.amax(pcd[:, 2])
    min_z = np.amin(pcd[:, 2])
    # xyz = pcd.copy()
    
    xyz_split = [pcd.copy()]

    for xyz in xyz_split:
        iters = 0
        while(iters < 25):
            plane = pyrsc.Plane()
            eq, inliers = plane.fit(xyz, thresh=0.01, minPoints=10000, maxIteration=100)
            if (inliers.shape[0] > 10000):
                planes.append(np.array([eq, xyz[inliers]], dtype=object))
                xyz = np.delete(xyz, inliers, axis=0)
                print(eq)
                iters = 0
            else:
                iters += 1

    return np.array(planes, dtype=object)

def split_planes_xyz(planes):
    x_planes = []
    y_planes = []
    z_planes = []
    for plane in planes:
        normal = plane[0][:3]
        if abs(normal[0]) > max(abs(normal[1]), abs(normal[2])):
            x_planes.append(plane)
        elif abs(normal[1]) > max(abs(normal[0]), abs(normal[2])):
            y_planes.append(plane)
        else:
            z_planes.append(plane)
        
    return x_planes, y_planes, z_planes

def delete_unfit_planes(planes, ref):
    delete_list = []
    for i in range(len(planes)):
        normal = planes[i][0][:3]
        delta = np.arccos(max(min(np.dot(normal, ref), 1), -1)) / np.pi
        if min(delta, 1 - delta) > 0.01:
            delete_list.append(i)
    planes = [i for j, i in enumerate(planes) if j not in delete_list]
    return planes

def find_layout_points(ceiling, x_planes, y_planes):
    layout_points = []
    # p0_coef = np.array(ceiling[0][:3])
    p0_coef = np.array([0, 0, np.round(ceiling[0][2])])
    p0_const = -ceiling[0][3]
    for i in range(len(x_planes)):
        p1 = x_planes[i]
        p1_coef = np.array([np.round(p1[0][0]), 0, 0])
        # p1_coef = np.array(p1[0][:3])
        p1_const = -p1[0][3]
        for j in range(len(y_planes)):
            # calculate intersection
            p2 = y_planes[j]
            p2_coef = np.array([0, np.round(p2[0][1]), 0])
            # p2_coef = np.array(p2[0][:3])
            p2_const = -p2[0][3]

            coef = np.array([p0_coef, p1_coef, p2_coef])
            const = np.array([p0_const, p1_const, p2_const])
            A = np.linalg.solve(coef, const)
            layout_points.append(A)

    return np.array(layout_points)

def find_layout_planes(planes):
    #TODO: split planes into xyz planes
    x_planes, y_planes, z_planes = split_planes_xyz(planes)

    #TODO: correct/remove plane normal
    x_planes = delete_unfit_planes(x_planes, np.array([1, 0, 0]))
    y_planes = delete_unfit_planes(y_planes, np.array([0, 1, 0]))
    z_planes = delete_unfit_planes(z_planes, np.array([0, 0, 1]))

    #TODO: find ceiling/floor
    ceiling = min(z_planes, key=lambda plane: plane[0][3] * plane[0][2])
    floor = max(z_planes, key=lambda plane: plane[0][3] * plane[0][2])

    #TODO: make grid
    layout_points = find_layout_points(ceiling, x_planes, y_planes)

    #TODO: project pcd to fill grid
    # print(np.round(ceiling[0][2]) * -ceiling[0][3])
    # print(np.round(floor[0][2]) * -floor[0][3])
    floor_height = np.round(floor[0][2]) * -floor[0][3]
    ceiling_height = np.round(ceiling[0][2]) * -ceiling[0][3]
    camera_height = 1.2704967260360718 #p06
    # camera_height = 1.2794110774993896 #p07
    floor_map = np.zeros((512, 512)) # 512 * 512, 160 FOV
    floor_map_range = (ceiling_height - camera_height) * np.tan(np.radians(80))
    floor_map_center = np.array([1.465893030166626, 5.175340175628662]) #p06
    # floor_map_center = np.array([1.706714153289795, 7.035895347595215]) #p07

    step = floor_map_range / 256 # meters per pixel
    pnt_map, u, c = convert_binary_map(layout_points[:, :2], floor_map_center, step, (512, 512))
    pcd_map, u, c = convert_binary_map(pcd[:, :2], floor_map_center, step, (512, 512))

    # for i in range(len(u)):
    #     if c[i] > 1:
    #         pcd_map[u[i][0], u[i][1]] = c[i]
    
    pnt_idx = np.where(pnt_map == 1)
    for i in range(len(pnt_idx[0]) - 1):
        y0 = pnt_idx[0][i]
        y1 = pnt_idx[0][i + 1] + 1
        for j in range(len(pnt_idx[1]) - 1):
            x0 = pnt_idx[1][j]
            x1 = pnt_idx[1][j + 1] + 1
            if np.sum(pcd_map[y0:y1, x0:x1]) / ((y1 - y0) * (x1 - x0)) > 0.9:
                floor_map[y0:y1, x0:x1] = 1
            # floor_map[y0, :] = 1
            # floor_map[:, x0] = 1
            # floor_map[y1, :] = 1
            # floor_map[:, x1] = 1