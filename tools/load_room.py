import os
import sys
import json
import trimesh

sys.path.append('D:/Gibson/3DSceneGraph/source/3DSceneGraph')
import main as graph_gen
from attributes import room_attributes, object_attributes, relationships

sys.path.append('D:/Gibson/3DSceneGraph/source/multiview_consistency')
from model import *

from load import *

def export_individual_room_inst_obj(building, gibson_mesh_path, export_mesh_path):
    ''' Export individual wavefront files of every room
        building            : the building class populated with 3D Scene graph values for the specific model to export
        gibson_mesh_path    : system path to Gibson dataset's raw mesh files (loads mesh.obj files)
        export_mesh_path    : system path to export the wavefront file
        filename            : name of the export file
    '''
    mesh = trimesh.load(gibson_mesh_path)
    colors=load_palette(palette_path)
    filename = building.name
    layer_type = building.room

    ## export OBJ file ##
    for layer in layer_type:
        file = open(os.path.join(export_mesh_path, f'{filename}_room_{layer_type[layer].id}.obj'), "w")
        #save vertices
        v_ = 'v '
        v_ += trimesh.util.array_to_string(mesh.vertices,col_delim=' ',row_delim='\nv ',digits=8) + '\n'
        file.write(v_)

        # add faces that are attributed an instance
        file.write("g Mesh\n")
        faces=mesh.faces[layer_type[layer].inst_segmentation,:]
        for face in faces:
            file.write("f ")
            for f in face:
                file.write(str(f + 1) + " ")
            file.write("\n")

    file.close()

def export_individual_room_inst_obj_clean(building, gibson_mesh_path, export_mesh_path):
    ''' Export individual wavefront files of every room
        building            : the building class populated with 3D Scene graph values for the specific model to export
        gibson_mesh_path    : system path to Gibson dataset's raw mesh files (loads mesh.obj files)
        export_mesh_path    : system path to export the wavefront file
        filename            : name of the export file
    '''
    mesh = trimesh.load(gibson_mesh_path)
    colors=load_palette(palette_path)
    filename = building.name

    ## export OBJ file ##
    for layer in building.room:
        file = open(os.path.join(export_mesh_path, f'{filename}_room_{building.room[layer].id}_clean.obj'), "w")
        #save vertices
        v_ = 'v '
        v_ += trimesh.util.array_to_string(mesh.vertices,col_delim=' ',row_delim='\nv ',digits=8) + '\n'
        file.write(v_)

        # add faces that are attributed an instance
        file.write("g Mesh\n")
        face_indices = building.room[layer].inst_segmentation

        room_elements = relationships.get_all_elemIDs(building.room[layer], building.object, building.room)
        for element in room_elements:
            print(len(face_indices))
            face_indices = np.delete(face_indices, np.in1d(face_indices, building.object[element].inst_segmentation).nonzero()[0], axis=0)
            print(len(face_indices))

        print('------------------')
        faces = mesh.faces[face_indices, :]
        for face in faces:
            file.write("f ")
            for f in face:
                file.write(str(f + 1) + " ")
            file.write("\n")

    file.close()

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
    pano_to_room, room_to_pano, pano_poses = cluster_pano(gibson_mesh_path, model, scenegraph3d[model]['graph'])
    # print_graph(scenegraph3d[model]['graph'])
    print(f'Processing {model}')

    if not os.path.exists(export_viz_path):
            os.makedirs(export_viz_path)

    # room_category = {}
    # for id in scenegraph3d[model]['graph'].room.keys():
    #     # ignore corridor
    #     room_category[id] = scenegraph3d[model]['graph'].room[id].scene_category

    # with open(os.path.join(opt.export_viz_path, f'{model}_clustering.json'), 'w') as outfile:
    #     json.dump(room_to_pano, outfile)
    
    # with open(os.path.join(opt.export_viz_path, f'{model}_category.json'), 'w') as outfile:
    #     json.dump(room_category, outfile)

    #TODO: generate room_inst_segmentation models (.obj)
    export_individual_room_inst_obj(scenegraph3d[model]['graph'], os.path.join(gibson_mesh_path, model, 'mesh.obj'), export_viz_path)
    export_individual_room_inst_obj_clean(scenegraph3d[model]['graph'], os.path.join(gibson_mesh_path, model, 'mesh.obj'), export_viz_path)

    # #TODO: load model
    # model_ = Model(os.path.join(gibson_mesh_path, model), None, model, None)
    
    # #TODO: re-sample points on 3D mesh surfaces
    # model_.num_sampled = 50  # number of points to sample per surface
    # model_.sampled_pnts = loader.sample_faces(model_.mesh_.vertices, model_.mesh_.faces, num_samples=int(model_.num_sampled))

    # #TODO: Generate room point cloud
    # room_path = export_viz_path
    # room = 'Allensville_room_4.obj'
    # face_inds, sampledPnts = graph_gen.find_room_span(os.path.join(room_path, room), model_.mesh_.vertices,
    #                                                 model_.mesh_.faces, model_.sampled_pnts, int(model_.num_sampled), \
    #                                                 room_path, override=True)  # load room's mesh

