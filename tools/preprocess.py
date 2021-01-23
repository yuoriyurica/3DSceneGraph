import os
import math
import json
import numpy as np
from PIL import Image

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    # assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def load_clustering(gibson_mesh_path, room_path, model):
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

if __name__ == '__main__':
    gibson_mesh_path = 'D:/Gibson/gibson_tiny'
    scene_graph_path = 'D:/Gibson/gibson_parse/layout'
    room_seg_path = 'D:/Gibson/gibson_parse/room_seg_model'
    models = 'Allensville Beechwood Benevolence Coffeen Collierville Corozal Cosmos Darden Forkland Hanson Hiteman Ihlen Klickitat Lakeville Leonardo Lindenwood Markleeville Marstons McDade Merom Mifflinburg Muleshoe Newfields Noxapater Onaga Pinesdale Pomaria Ranchester Shelbyville Stockman Tolstoy Uvalda Wainscott Wiconisco Woodbine'
    for model in models.split(' '):

        filePath = os.path.join('D:/Gibson/gibson_parse/pano_mah_aligned', model)

        room_path = os.path.join(room_seg_path, model)
        room_to_pano = load_clustering(gibson_mesh_path, room_path, model)
        
        for room_id in room_to_pano.keys():
            layout_mat = np.loadtxt(os.path.join('D:/Gibson/gibson_parse/pano_mah_aligned', model, f'{room_to_pano[room_id][0]}_aligned_VP.txt'), dtype=float)[2::-1]
            for pano in room_to_pano[room_id]:
                mat = np.loadtxt(os.path.join('D:/Gibson/gibson_parse/pano_mah_aligned', model, f'{pano}_aligned_VP.txt'), dtype=float)[2::-1]
                rot = rotationMatrixToEulerAngles(mat)[2] - rotationMatrixToEulerAngles(layout_mat)[2]

                print(rot)



        # subfilePath = os.listdir(filePath)
        # for file in subfilePath:
        #     folderPath = os.path.join(filePath, file)

        #     if os.path.isfile(folderPath):
        #         continue

            
            