project_path=/d/Gibson/3DSceneGraph                   # system path to the 3D Scene Graph folder
file_path=$project_path/tools/load_mesh_layout.py           # system path to loading function
# file_path=$project_path/tools/fix_error_layout.py           # system path to loading function
verified=1                                      # 0 or 1 depending on whether to load only automated or verified results
visualize=0                                     # 0 or 1 depending on whether to export or not wavefront files with segmentation
data_path=/d/Gibson/3DSceneGraph_tiny/data                    # system path to 3D Scene Graph results
palette_path=$project_path/tools/palette.txt    # system path to color palette (for visualization purposes)
gibson_mesh_path=/d/Gibson/gibson_tiny       # system path to Gibson database model data
export_viz_path=$project_path/visuals           # system path to export wavefront files (if visualize is set to 1)

tiny_models='Allensville Beechwood Benevolence Coffeen Collierville Corozal Cosmos Darden Forkland Hanson Hiteman Ihlen Klickitat Lakeville Leonardo Lindenwood Markleeville Marstons McDade Merom Mifflinburg Muleshoe Newfields Noxapater Onaga Pinesdale Pomaria Ranchester Shelbyville Stockman Tolstoy Uvalda Wainscott Wiconisco Woodbine'
medium_models='Adairsville Airport Albertville Anaheim Ancor Andover Annona Arkansaw Athens Bautista Bohemia Bonesteel Bonnie Brinnon Broseley Brown Browntown Byers Castor Cauthron Chilhowie Churchton Clairton Cochranton Cottonport Couderay Cousins Darrtown Donaldson Duarte Eagan Edson Emmaus Frankfort German Globe Goffs Goodfield Goodwine Goodyear Gravelly Hainesburg Helton Highspire Hildebran Hillsdale Hominy Hordville Hortense Irvine Kemblesville Kobuk Losantville Lynchburg Maida Marland Martinville Maugansville Micanopy Musicks Natural Neibert Neshkoro Newcomb Nuevo Oyens Pablo Pamelia Parole Pearce Pittsburg Poipu Potterville Readsboro Rockport Rogue Rosser Sands Scioto Shelbiana Silas Soldier Southfield Springerville Stilwell Sugarville Sunshine Sussex Sweatman Swisshome Swormville Thrall Tilghmanton Timberon Tokeland Touhy Tyler Victorville Waipahu Wando Westfield Willow Wilseyville Winooski Wyldwood'

test_models='Allensville'

#iterate over models. Choose: tiny_models for tiny split or medium_models for medium split
# for model in $tiny_models; do
#     python $file_path --model $model \
#                     --verified $verified \
#                     --visualize $visualize \
#                     --data_path $data_path \
#                     --palette_path $palette_path \
#                     --gibson_mesh_path $gibson_mesh_path \
#                     --export_viz_path $export_viz_path
# done

verified=0
gibson_mesh_path=/d/Gibson/gibson_medium
data_path=/d/Gibson/3DSceneGraph_medium/data

for model in $medium_models; do
    ((i=i%8)); ((i++==0)) && wait
    python $file_path --model $model \
                    --verified $verified \
                    --visualize $visualize \
                    --data_path $data_path \
                    --palette_path $palette_path \
                    --gibson_mesh_path $gibson_mesh_path \
                    --export_viz_path $export_viz_path &
done

wait
