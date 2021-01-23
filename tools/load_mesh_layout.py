import os
import sys
import trimesh
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import decomposition
from collections import deque

sys.path.append('D:/Gibson/3DSceneGraph/source/3DSceneGraph')
import main as graph_gen
from attributes import room_attributes

sys.path.append('D:/Gibson/3DSceneGraph/source/multiview_consistency')
from model import *

from load import *
from layout_tools.utils import *

class Plane():
    ''' Plane attributes '''
    def __init__(self):
        self.faces_idx = None
        self.faces = None
        self.area = None
        self.normals = None
        self.normal = None
        self.vertices = None
        self.points_idx = None
        self.points = None
    
    def set_attribute(self, attribute, value):
        ''' Set a camera attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown camera attribute: {}'.format(attribute))
            return
        self.__dict__[attribute] = value
     
    def get_attribute(self, attribute):
        ''' Get a camera attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown camera attribute: {}'.format(attribute))
            return -1
        return self.__dict__[attribute]

def group_faces(mesh, log=False):
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
            
            if log:
                print(f'{len(visited)} / {len(graph)}')

    return planes

def check_axis_correlate(normal, axis, threshold=0.01):
    ref = np.array([0, 0, 0])
    ref[axis] = 1

    delta = np.arccos(max(min(np.dot(normal, ref), 1), -1)) / np.pi
    if min(delta, 1 - delta) < threshold:
        return True
    else:
        return False

def unpack_plane(mesh_, faces_idx):
    faces = mesh_.faces[faces_idx]
    area = sum(mesh_.area_faces[faces_idx])
    normals = mesh_.face_normals[faces_idx]
    vertices = mesh_.vertices[faces]

    # if area > 0.4:
    # mesh_.visual.face_colors[faces_idx] = trimesh.visual.random_color()
    
    normal = np.average(normals, axis=0)

    points_idx = np.unique(faces)
    points = mesh_.vertices[points_idx]

    new_plane = Plane()
    new_plane.set_attribute('faces_idx', faces_idx)
    new_plane.set_attribute('faces', faces)
    new_plane.set_attribute('area', area)
    new_plane.set_attribute('normals', normals)
    new_plane.set_attribute('normal', normal)
    new_plane.set_attribute('vertices', vertices)
    new_plane.set_attribute('points_idx', points_idx)
    new_plane.set_attribute('points', points)

    return new_plane

def unpack_all_planes(mesh_, planes_):
    planes = []
    for key in planes_.keys():
        faces_idx = list(planes_[key])
        planes.append(unpack_plane(mesh_, faces_idx))
    
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

def get_neighbor_contours(i, cnts, return_index=True):
    if return_index:
        if i == 0:
            pi = len(cnts) - 1
            ni = i + 1
        elif i == len(cnts) - 1:
            pi = i - 1
            ni = 0
        else:
            pi = i - 1
            ni = i + 1

        return pi, ni

def get_neighbor_edges(i, cnts):
    pi, ni = get_neighbor_contours(i, cnts)
    e0 = cnts[pi] - cnts[i]
    e1 = cnts[ni] - cnts[i]
    return e0, e1

def get_theta(rad):
    # if close to 0
    d0 = -rad

    # if close to pi/-pi
    if abs(np.pi - rad) < abs(-np.pi - rad):
        d1 = np.pi - rad
    else:
        d1 = -np.pi - rad

    # if close to pi/2
    d2 = np.pi / 2 - rad

    # if close to -pi/2
    d3 = -np.pi / 2 - rad

    d = np.array([d0, d1, d2, d3])
    return d[np.argmin(abs(d))]

def check_align(cnts):
    edges = [(cnts[i], cnts[i+1]) for i in range(len(cnts) - 1)]
    edges.append((cnts[len(cnts) - 1], cnts[0]))

    edges_length = [np.linalg.norm(edge[1] - edge[0]) for edge in edges]
    threshold = np.mean(edges_length)
    # threshold = np.percentile(edges_length, 75)

    candidate = {}
    candidate_length = {}
    for i in range(len(edges)):
        e0 = edges[i][1] - edges[i][0]

        if np.linalg.norm(e0) < threshold:
            continue
        
        candidate[i] = [edges[i]]
        candidate_length[i] = edges_length[i]

        for j in range(len(edges)):
            if i == j:
                continue
            
            e1 = edges[j][1] - edges[j][0]
            delta = abs(np.arccos(np.dot(e0, e1) / (np.linalg.norm(e0) * np.linalg.norm(e1))) / np.pi - 0.5)
            
            if delta < 0.02 or delta > 0.48:
                candidate[i].append(edges[j])
                candidate_length[i] += edges_length[j]

    largest_candidate = max(candidate.keys(), key=lambda x: candidate_length[x])

    # rotate clockwise to find rotate angles
    for edge in candidate[largest_candidate]:
        x = edge[1][1] - edge[0][1]
        y = edge[1][0] - edge[0][0]
        rad = np.arctan2(y, x)
        deg = np.round(rad * 180 / np.pi).astype(int)
        print(deg % -90)

        result = np.zeros((512, 512, 3))
        cv2.line(result, tuple(edge[0]), tuple(edge[1]), (255, 0, 0), 1)
        cv2.imshow('window', result)
        cv2.waitKey()

        if abs(deg % -90) < abs(deg % 90):
            rad = np.radians(deg % -90)
        else:
            rad = np.radians(deg % 90)

        matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        vec = np.round(np.matmul(matrix, np.array([y, x]))).astype(int)

        cv2.line(result, tuple(edge[0]), tuple(edge[0] + vec), (255, 0, 0), 1)
        cv2.imshow('window', result)
        cv2.waitKey()
        cv2.destroyAllWindows()

    result = np.zeros((512, 512, 3))
    for edge in candidate[largest_candidate]:
        cv2.line(result, tuple(edge[0]), tuple(edge[1]), (255, 0, 0), 1)
    
    for cnt in cnts:
        cv2.circle(result, tuple(cnt), 2, (0, 255, 0), 1)

    cv2.imshow('window', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # For Debugging
    # while(len(candidate.keys()) != 0):

    #     largest_candidate = max(candidate.keys(), key=lambda x: candidate_length[x])

    #     result = np.zeros((512, 512, 3))
    #     for edge in candidate[largest_candidate]:
    #         cv2.line(result, tuple(edge[0]), tuple(edge[1]), (255, 255, 255), 1)
        
    #     for cnt in cnts:
    #         cv2.circle(result, tuple(cnt), 2, (0, 255, 0), 1)

    #     cv2.imshow('window', result)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    #     del candidate[largest_candidate]

def correct_contours(cnts, interactive=False):
    meta = {}
    pivot = 0
    edge_pivot = 1
    for i in range(len(cnts)):
        e0, e1 = get_neighbor_edges(i, cnts)
        max_edge = np.linalg.norm(e0) + np.linalg.norm(e1)

        pivot_e0, pivot_e1 = get_neighbor_edges(pivot, cnts)
        max_pivot_edge = np.linalg.norm(pivot_e0) + np.linalg.norm(pivot_e1)
        
        delta = abs(np.arccos(np.dot(e0, e1) / (np.linalg.norm(e0) * np.linalg.norm(e1))) / np.pi - 0.5)
        # print(i, delta, max_edge)

        # if delta < 0.02 and max_pivot_edge <= max_edge:
        if max_pivot_edge <= max_edge:
            pivot = i
            if np.linalg.norm(e0) < np.linalg.norm(e1):
                _, edge_pivot = get_neighbor_contours(i, cnts)
            else:
                edge_pivot, _ = get_neighbor_contours(i, cnts)

    raw_cnts = cnts.copy()
    last_i = None
    next_j = None
    cnt_seq = list(range(len(cnts)))[pivot:] + list(range(len(cnts)))[:pivot]
    for i in range(len(cnts)):
        if i == len(cnts) - 1:
            j = cnt_seq[0]
            i = cnt_seq[i]
        else:
            j = cnt_seq[i + 1]
            i = cnt_seq[i]

        if j == pivot:
            if cnts[next_j, 0] != cnts[j, 0]:
                cnts[i, 0] = cnts[j, 0]
                cnts[i, 1] = cnts[last_i, 1]
            elif cnts[next_j, 1] != cnts[j, 1]:
                cnts[i, 1] = cnts[j, 1]
                cnts[i, 0] = cnts[last_i, 0]
            elif cnts[i, 0] != cnts[j, 0] and cnts[i, 1] != cnts[j, 1]:
                d0 = cnts[j, 0] - cnts[i, 0]
                d1 = cnts[j, 1] - cnts[i, 1]
                # print(d0, d1)
                if abs(d0) < abs(d1):
                    cnts[j, 0] = cnts[i, 0]
                else:
                    cnts[j, 1] = cnts[i, 1]
        else:
            if cnts[i, 0] != cnts[j, 0] and cnts[i, 1] != cnts[j, 1]:
                d0 = cnts[j, 0] - cnts[i, 0]
                d1 = cnts[j, 1] - cnts[i, 1]
                # print(d0, d1)
                if abs(d0) < abs(d1):
                    cnts[j, 0] = cnts[i, 0]
                else:
                    cnts[j, 1] = cnts[i, 1]
                
            last_i = i

        if i == pivot:
            next_j = j

        if interactive:
            floor_map = cv2.drawContours(np.zeros((res, res)), [cnts], -1, (1,), -1)
            corner_result = np.stack([floor_map, floor_map, floor_map], axis=2)
            for k in range(len(cnts)):
                if k == pivot:
                    cv2.circle(corner_result, (int(cnts[k, 0]), int(cnts[k, 1])), 7, (0, 0, 255), 2)
                elif k == i:
                    cv2.circle(corner_result, (int(cnts[k, 0]), int(cnts[k, 1])), 7, (255, 0, 0), 2)
                elif k == j:
                    cv2.circle(corner_result, (int(cnts[k, 0]), int(cnts[k, 1])), 7, (0, 255, 0), 2)
                else:
                    cv2.circle(corner_result, (int(cnts[k, 0]), int(cnts[k, 1])), 7, (0, 255, 0), 2)
            cv2.imshow('floor_map', corner_result)
            cv2.waitKey()

    corner_pre = cv2.drawContours(np.zeros((res, res)), [raw_cnts], -1, (255,), -1)
    corner_post = cv2.drawContours(np.zeros((res, res)), [cnts], -1, (255,), -1)
    corner_result = np.dstack((corner_pre, corner_post, corner_post))
    for i in range(len(cnts)):
        # cv2.circle(corner_result, (int(raw_cnts[i, 0]), int(raw_cnts[i, 1])), 7, (255, 0, 0), 2)
        cv2.circle(corner_result, (int(cnts[i, 0]), int(cnts[i, 1])), 7, (0, 255, 0), 2)
        
    return cnts, corner_result

def read_mah_mat(model, pano):
    path = os.path.join(f'D:\Gibson\gibson_parse\pano_mah_aligned\{model}', f'{pano}_aligned_VP.txt')
    return np.loadtxt(path, dtype=float)[2::-1]

def load_clustering(gibson_mesh_path, room_path, model, data_path):
    clustering_file_path = os.path.join(room_path, f'{model}_clustering.npy')
    if not os.path.exists(clustering_file_path):
        # load the 3D Scene Graph data. 
        scenegraph3d = {}
        scenegraph3d[model] = {}
        scenegraph3d[model]['graph'], scenegraph3d[model]['panoramas'] = load_3DSceneGraph(model, data_path)
        pano_to_room, room_to_pano, pano_poses = cluster_pano(gibson_mesh_path, model, scenegraph3d[model]['graph'])
        np.save(clustering_file_path, np.array(room_to_pano, dtype=object))
        print(f'Saved')
    else:
        room_to_pano = np.load(clustering_file_path, allow_pickle=True).item()
    
    return room_to_pano

def load_mesh(obj_path):
    mesh_ = trimesh.load(obj_path)
    matrix = Model.Rx(None, 90) 
    mesh_.apply_transform(matrix)

    return mesh_

def load_grouping(grouping_file_path, mesh_, force_regroup=False):
    if not os.path.exists(grouping_file_path) or force_regroup:
        planes_= group_faces(mesh_)
        np.save(grouping_file_path, np.array(planes_, dtype=object))
        print(f'Saved')
    else:
        planes_ = np.load(grouping_file_path, allow_pickle=True).item()

    return planes_

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

    room_path = os.path.join('D:/Gibson/gibson_parse/room_seg_model', model)
    room_to_pano = load_clustering(gibson_mesh_path, room_path, model, data_path)

    if not os.path.exists(export_viz_path):
            os.makedirs(export_viz_path)

    for room_id in room_to_pano.keys():
    # for room_id in [9]:
        print(f'processing {model} room {room_id}')
        is_clean = '_clean'
        export_path = os.path.join('D:/Gibson/gibson_parse/layout', model, f'{model}_room_{room_id}')

        #TODO: skip processed layout
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        # else:
        #     continue
        
        #TODO: load model
        obj_path = os.path.join(room_path, f'{model}_room_{room_id}{is_clean}.obj')  # file path to mesh model (.obj)
        mesh_ = load_mesh(obj_path)

        #TODO: load face grouping
        grouping_file_path = os.path.join(room_path, f'{model}_room_{room_id}_face_grouping.npy')
        planes_ = load_grouping(grouping_file_path, mesh_)

        # select reference view
        meta = {}
        if len(room_to_pano[room_id]) != 0:
            _, camera_pos = get_point(gibson_mesh_path, model, room_to_pano[room_id][0])
            camera_height = camera_pos[2]
            floor_map_center = camera_pos[:2]

            # apply manhattan align matrix
            mah_mat = np.zeros((4, 4))
            mah_mat[3][3] = 1
            mah_mat[:3, :3] = read_mah_mat(model, room_to_pano[room_id][0])
            
            inv_transl_mat = np.array([[1, 0, 0, -camera_pos[0]], [0, 1, 0, -camera_pos[1]], [0, 0, 1, -camera_pos[2]], [0, 0, 0, 1]])
            transl_mat = np.array([[1, 0, 0, camera_pos[0]], [0, 1, 0, camera_pos[1]], [0, 0, 1, camera_pos[2]], [0, 0, 0, 1]])
            mesh_.apply_transform(inv_transl_mat)
            mesh_.apply_transform(mah_mat)
            mesh_.apply_transform(transl_mat)
            meta['view'] = room_to_pano[room_id]
            meta['rot'] = mah_mat.tolist()
        else:
            camera_height = 1.6 # default setting
            floor_map_center = np.array([(np.amin(ceiling.points[:, 0]) + np.amax(ceiling.points[:, 0])) / 2, (np.amin(ceiling.points[:, 1]) + np.amax(ceiling.points[:, 1])) / 2])
            meta['view'] = None
            meta['rot'] = None

        # print(f'num planes: {len(planes_)}')
        planes = unpack_all_planes(mesh_, planes_)

        try:
            ceiling = max(planes, key=lambda p: p.area * -p.normal[2] if np.all(p.points[:, 2] > camera_height) else -np.inf)
            floor = max(planes, key=lambda p: p.area if np.all(p.points[:, 2] < camera_height) else -np.inf)
            # ceiling = min(planes, key=lambda plane: plane[0][3] * len(plane[2]) if plane[0][2] < 0 else np.inf)
            # floor = max(planes, key=lambda plane: plane[0][3] / len(plane[2]) if plane[0][2] > 0 else -np.inf)
        except Exception as e:
            print(f'Error in ceiling selection: {e}')
            with open('error_cases.txt', 'a') as out_file:
                out_file.write(f'{model} room {room_id}: Error in ceiling selection: {e}\n')

            continue
        
        # mesh_.visual.face_colors[ceiling.faces_idx] = trimesh.visual.random_color()
        # mesh_.show()
        # exit()

        # set floormap parameters
        ceiling_height = min(ceiling.points, key=lambda p: p[2])[2]
        floor_height = min(floor.points, key=lambda p: p[2])[2]
        floor_map_range = (ceiling_height - camera_height) * np.tan(np.radians(80))
        res = 512
        step = floor_map_range / (res / 2) # meters per pixel

        # convert to opencv coordinates
        face_ij = convert_ij(ceiling.vertices[:, :, :2], floor_map_center, step, res)

        # find contours
        try:
            pcd_map = np.zeros((res, res)) # 512 * 512, 160 FOV
            cv2.fillPoly(pcd_map, list(face_ij), (1,))
            cnts, corner_result = detect_contours(pcd_map, return_map='corner')

            # check_align(cnts)
            # continue

            cnts, corner_map = correct_contours(cnts, interactive=False)
            
            floor_map = cv2.drawContours(np.zeros((res, res)), [cnts], -1, (1,), -1)
        except Exception as e:
            print(f'Error in contour detection: {e}')
            with open('error_cases.txt', 'a') as out_file:
                out_file.write(f'{model} room {room_id}: Error in contour detection: {e}\n')
            continue

        # calculate IoU  
        try:      
            overlap = pcd_map.astype(bool) * floor_map.astype(bool) # Logical AND
            union = pcd_map.astype(bool) + floor_map.astype(bool) # Logical OR

            IOU = overlap.sum() / float(union.sum())
            meta['IOU'] = IOU
            print(f'IOU: {IOU}')
        except Exception as e:
            print(f'Error in IoU calculation: {e}')
            with open('error_cases.txt', 'a') as out_file:
                out_file.write(f'{model} room {room_id}: Error in IoU calculation: {e}\n')
            continue
        
        if IOU > 0.9:
            cv2.imwrite(os.path.join(export_path, f'floormap.jpg'), floor_map * 255)
            cv2.imwrite(os.path.join(export_path, f'floormap_corner.png'), corner_map * 255)
            cv2.imwrite(os.path.join(export_path, f'pcdmap.jpg'), pcd_map * 255)

            meta['ceiling_height'] = ceiling_height
            meta['floor_height'] = floor_height
            meta['camera_height'] = camera_height
            meta['floor_map_center'] = floor_map_center.tolist()
            meta['floor_map_range'] = floor_map_range
            meta['res'] = 512
            meta['step'] = step
            save_json(os.path.join(export_path, f'meta.json'), meta)

    

