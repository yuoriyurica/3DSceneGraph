import os
import sys
import trimesh
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import linear_model
from collections import deque

sys.path.append('D:/Gibson/3DSceneGraph/source/3DSceneGraph')
import main as graph_gen
from attributes import room_attributes

sys.path.append('D:/Gibson/3DSceneGraph/source/multiview_consistency')
from model import *

from load import *
from layout_tools.utils import *

def group_faces(mesh):
    graph = nx.from_edgelist(mesh.face_adjacency)

    #TODO: 1. Go through all faces in one-ring order (BFS)
    #TODO: 2. Check face normals: split into new group/graph if not parallel
    visited = set()
    not_visited = set(graph.nodes())
    planes = {}
    while len(not_visited) != 0:
        root = None
        q = deque()
        q.append(not_visited.pop())
        while q:
            node = q.popleft()

            if root is None:
                planes[node] = set((node,))
                visited.add(node)
                # print(f'set {root} as root')
                node_not_visited = set(graph[node].keys()) - visited
                list(map(q.append, node_not_visited))
                list(map(visited.add, node_not_visited))
                list(map(not_visited.remove, node_not_visited))
                root = node

            else:
                ref = mesh.face_normals[root]
                normal = mesh.face_normals[node]
                delta = np.arccos(max(min(np.dot(normal, ref), 1), -1)) / np.pi

                if min(delta, 1 - delta) < 0.1:
                    planes[root].add(node)
                    visited.add(node)
                    # print(f'add {node} to set {root}')
                    node_not_visited = set(graph[node].keys()) - visited
                    list(map(q.append, node_not_visited))
                    list(map(visited.add, node_not_visited))
                    list(map(not_visited.remove, node_not_visited))
                else:
                    visited.remove(node)
                    not_visited.add(node)
                    # print(f'discard {node}')
            
            print(f'{len(visited)} / {len(graph)}')

    return planes

def medfit(data):
    groups = np.array_split(data, 3)
    group_median = []
    for group in groups:
        group_median.append(np.median(group, axis=0))
    return np.average(group_median, axis=0)

def check_axis_correlate(normal, axis, threshold=0.01):
    ref = np.array([0, 0, 0])
    ref[axis] = 1

    delta = np.arccos(max(min(np.dot(normal, ref), 1), -1)) / np.pi
    if min(delta, 1 - delta) < threshold:
        return True
    else:
        return False

def fill_faces(pnt_map, res):
    floor_map = np.zeros((res, res)) # 512 * 512, 160 FOV
    
    pnt_idx = np.where(pnt_map == 1)
    pnt_idx = np.column_stack((pnt_idx[1], pnt_idx[0]))
    cv2.polylines(floor_map, [pnt_idx.reshape((-1,1,2))], False, (1,), 1)
    cv2.fillPoly(floor_map, [pnt_idx.reshape((-1,1,2))], (1,))

    pnt_idx = pnt_idx[pnt_idx[:, 1].argsort()]
    pnt_idx = pnt_idx[pnt_idx[:, 0].argsort(kind='mergesort')]
    cv2.polylines(floor_map, [pnt_idx.reshape((-1,1,2))], False, (1,), 1)
    cv2.fillPoly(floor_map, [pnt_idx.reshape((-1,1,2))], (1,))

    return floor_map

def filter_small_planes(mesh_, planes_, threshold=1000):
    planes = []
    for key in planes_.keys():
        face = mesh_.faces[key]
        faces = np.unique(mesh_.faces[list(planes_[key])])
        normal = mesh_.face_normals[key]
        origin = mesh_.vertices[face][0]

        # mesh_.visual.face_colors[list(planes_[key])] = trimesh.visual.random_color()

        normals = mesh_.face_normals[list(planes_[key])]
        # normal = np.median(normals, axis=0)
        normal = medfit(normals)

        origins = mesh_.vertices[faces].reshape((1, -1, 3))[0]
        # origin = np.median(origins, axis=0)
        origin = medfit(origins)

        d = np.dot(normal, -origin)
        if len(planes_[key]) > threshold:
            planes.append([np.array([*normal, d]), mesh_.vertices[faces], list(planes_[key])])
    
    return planes

def detect_contours(pcd_map, return_map=None):
    floor_map = pcd_map * 255
    pcd_map_gray = np.uint8(floor_map)
    ret, thresh = cv2.threshold(pcd_map_gray, 127, 255, 0)
    cnts, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_cnts = np.array(cnts[0]).transpose((1, 0, 2))[0]

    epsilon = cv2.arcLength(cnts[0], True) * 0.005
    approx = cv2.approxPolyDP(cnts[0], epsilon, True) # thresholding exist here
    cnts = np.array(approx).transpose((1, 0, 2))[0]

    if return_map == 'corner':
        floor_map = cv2.drawContours(np.zeros((res, res)), [cnts], -1, (1,), -1)
        corner_result = np.stack([floor_map, floor_map, floor_map], axis=2)
        for i in range(len(cnts)):
            cv2.circle(corner_result, (int(cnts[i, 0]), int(cnts[i, 1])), 7, (0, 255, 0), 2)

        return cnts, corner_result
    elif return_map == 'raw':
        raw_corner_result = np.stack([floor_map, floor_map, floor_map], axis=2)
        for i in range(len(raw_cnts)):
            cv2.circle(raw_corner_result, (int(raw_cnts[i, 0]), int(raw_cnts[i, 1])), 7, (0, 255, 0), 2)

        return cnts, raw_corner_result
    else:
        return cnts

def correct_contours(cnts, interactive=False):
    for i in range(len(cnts)):
        if i == len(cnts) - 1:
            j = 0

            if cnts[i, 0] != cnts[j, 0] and cnts[i, 1] != cnts[j, 1]:
                d0 = cnts[j, 0] - cnts[i, 0]
                d1 = cnts[j, 1] - cnts[i, 1]
                if abs(d0) < abs(d1):
                    cnts[i, 0] = cnts[j, 0]
                else:
                    cnts[i, 1] = cnts[j, 1]
        else:    
            j = i + 1
            
            if cnts[i, 0] != cnts[j, 0] and cnts[i, 1] != cnts[j, 1]:
                d0 = cnts[j, 0] - cnts[i, 0]
                d1 = cnts[j, 1] - cnts[i, 1]
                print(d0, d1)
                if abs(d0) < abs(d1):
                    cnts[j, 0] = cnts[i, 0]
                else:
                    cnts[j, 1] = cnts[i, 1]

        if interactive:
            floor_map = cv2.drawContours(np.zeros((res, res)), [cnts], -1, (1,), -1)
            corner_result = np.stack([floor_map, floor_map, floor_map], axis=2)
            for i in range(len(cnts)):
                cv2.circle(corner_result, (int(cnts[i, 0]), int(cnts[i, 1])), 7, (0, 255, 0), 2)
            cv2.imshow('floor_map', corner_result)
            cv2.waitKey()

    return cnts

def load_layout():
    # reference values
    ceiling_height = 2.4207375713032206
    floor_height =  -0.032145283340840346
    print(ceiling_height, floor_height)

    layout = np.load('layout.npy', allow_pickle=True) # floormap
    layout_res = 1024
    layout_center = np.array([0.77414263, 6.21963634, 1.6]) # [x, y, ceil - cam]
    layout_map_range = (ceiling_height - layout_center[2]) * np.tan(np.radians(80))
    layout_step = layout_map_range / (layout_res / 2)
    
    # pano values
    pano = 'p07'

    if pano == 'p06':
        camera_height = 1.2704967260360718
        floor_map_center = np.array([1.465893030166626, 5.175340175628662])
    elif pano == 'p07':
        camera_height = 1.2794110774993896
        floor_map_center = np.array([1.706714153289795, 7.035895347595215])
    
    floor_map_range = (ceiling_height - camera_height) * np.tan(np.radians(80))
    res = 512
    step = floor_map_range / (res / 2) # meters per pixel
    
    layout_map_xy = convert_xy(np.vstack(np.where(layout == 1)).T, layout_center[:2], layout_step, layout_res)
    
    floor_map = convert_binary_map(layout_map_xy, floor_map_center, step, res)

    cv2.imwrite(f'{pano}_floormap_from_ref.jpg', floor_map * 255)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of Gibson database model")
    parser.add_argument("--verified", type=int, help="Boolean to define whether to load verified (1) or only automated (0) results")
    parser.add_argument("--visualize", type=int, help="Boolean to define whether to visualize (1) or not (0) the 3D Scene Graph mesh segmentation results")
    parser.add_argument("--data_path", type=str, help="System path to 3D Scene Graph data")
    parser.add_argument("--palette_path", type=str, default=None, help="System path to predefined color palette for visualizing")
    parser.add_argument("--gibson_mesh_path", type=str, default=None, help="System path to Gibson dataset's raw mesh files (loads mesh.obj files)")
    parser.add_argument("--export_viz_path", type=str, default=None, help="System path to export wavefront files when visualizing")
    opt = parser.parse_args()

    # define parameters
    model = opt.model
    if opt.verified:
        result_type = 'verified_graph'
    else:
        result_type = 'automated_graph'
    data_path = os.path.join(opt.data_path, result_type)
    gibson_mesh_path = opt.gibson_mesh_path
    export_viz_path = os.path.join(opt.export_viz_path, model)
    palette_path = opt.palette_path

    # load the 3D Scene Graph data. 
    # 'graph' : the 3D Scene Graph structured data
    # 'panoramas' : the projection of labels on the 2D panoramas (after as step of snapping to image boundaries with superpixels)
    scenegraph3d = {}
    scenegraph3d[model] = {}
    scenegraph3d[model]['graph'], scenegraph3d[model]['panoramas'] = load_3DSceneGraph(model, data_path)
    # print_graph(scenegraph3d[model]['graph'])

    if not os.path.exists(export_viz_path):
            os.makedirs(export_viz_path)

    #TODO: load model
    is_clean = '_clean'
    room_path = export_viz_path
    room_id = 4
    obj_path = os.path.join(room_path, f'{model}_room_{room_id}{is_clean}.obj')  # file path to mesh model (.obj)
    mesh_ = trimesh.load(obj_path)
    matrix = Model.Rx(None, 90) 
    mesh_.apply_transform(matrix)

    if 0:
        planes_= group_faces(mesh_)
        # np.save(f'BFS{is_clean}', np.array(planes_, dtype=object))
        # print(f'Saved')
    else:
        planes_ = np.load(f'BFS{is_clean}.npy', allow_pickle=True).item()
    
    print(f'plane count: {len(planes_)}')

    planes = filter_small_planes(mesh_, planes_)

    # because normal faces in, find ceiling and floor plane
    ceiling = min(planes, key=lambda plane: plane[0][3] * plane[0][2])
    floor = max(planes, key=lambda plane: plane[0][3] * plane[0][2] if abs(plane[0][2]) > 0.99 else -np.inf)
    
    # find vertices and faces of ceiling plane
    faces = mesh_.faces[ceiling[2]]
    edges = np.array([[[faces[i][0], faces[i][1]], [faces[i][1], faces[i][2]], [faces[i][2], faces[i][0]]] for i in range(len(faces))])
    pcd = mesh_.vertices[faces].reshape((1, -1, 3))[0]

    # set floormap parameters
    # FIXME: need to filter out small planes first (relatively large value)
    ceiling_height = np.round(ceiling[0][2]) * -ceiling[0][3]
    floor_height = np.round(floor[0][2]) * -floor[0][3]
    print(ceiling_height, floor_height)

    camera_height = 1.6 # default setting
    floor_map_center = np.array([(np.amin(pcd[:, 0]) + np.amax(pcd[:, 0])) / 2, (np.amin(pcd[:, 1]) + np.amax(pcd[:, 1])) / 2])
    floor_map_range = (ceiling_height - camera_height) * np.tan(np.radians(80))
    res = 1024
    step = floor_map_range / (res / 2) # meters per pixel

    # convert to floormap coordinates
    edge_vertices = convert_ij(mesh_.vertices[edges][:, :, :, :2], floor_map_center, step, res)[:, :, :, ::-1]
    face_vertices = convert_ij(mesh_.vertices[faces][:, :, :2], floor_map_center, step, res)[:, :, ::-1]

    # find contours
    pcd_map = np.zeros((res, res)) # 512 * 512, 160 FOV
    # cv2.polylines(pcd_map, edge_vertices.reshape(1, -1, 2, 2)[0], True, (1,), 1)
    cv2.fillPoly(pcd_map, [face_vertices[i] for i in range(len(face_vertices))], (1,))
    cnts = detect_contours(pcd_map)
    cnts = correct_contours(cnts, interactive=True)
    
    floor_map = cv2.drawContours(np.zeros((res, res)), [cnts], -1, (1,), -1)
    cv2.imwrite(os.path.join(room_path, f'{model}_room_{room_id}_floormap.jpg'), floor_map * 255)

    

