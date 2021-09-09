# Relation generation and Image rendering
This is code for relationship generation and image rendering.
## Relation generation 

### Environment

ubuntu==18.04

python==3.6

sapien==1.0

transforms3d

### Notice
Several important paths are needed in *relation_generate.py*, and you should make 
sure these paths are available.

-[x] visual_model_path:  e.g. ShapeNetCore.v2

-[x] collison_model_path: For shapenet models, they don't have collision models.
we use v-hacd algorithm to generate collision models with pybullet.(Please refer https://github.com/kmammou/v-hacd)


## Image Rendering
We utilize Blender to render the scene to make it more realistic. See *Blender_Renderer.py*

### Environment 
Blender==2.83

bpycv(see https://github.com/DIYer22/bpycv for install)

### notice
We use domain randomization technique to augment the rendered image. There are two significant
paths you should notice.

-[x] table_back_path: the table background.

-[x] world_back_path: blender world background. The format is '.hdr', you can download in https://hdrihaven.com/hdris/ and
store them in the ./world_background folder.