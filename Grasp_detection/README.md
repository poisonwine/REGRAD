#Grasp Detection
###Environment

1.ubuntu 16.04; 

2.python = 3.7;

3.Download "Blender-2.83.2-linux64" from the official website or from this link "https://download.blender.org/release/Blender2.83/blender-2.83.2-linux64.tar.xz";

4.Download "Blensor-1.0.18-RC10-x64" from the official website or from this two links "https://www.blensor.org/dload/blensor_1.0.18-RC10_x64.tbz", "https://www.blensor.org/dload/Blensor-x64.AppImage". Both of them should be downloaded.

5.Put the files "pcd_viewer" and "render_pointcloud" as well as the "Blensor-x64.AppImage" into the folder of "Blensor-1.0.18-RC10-x64".
### How to use 
1. **Get scaled objects from original model**
   ```
   ./blender-2.83.2-linux64/blender --python scale_objects.py
   ```
   
   @param: **input_model_dir** = "/your/model/path/obj" \
   @param: **input_info_dir** = "/your/model_info/path/*.json" \
   @param: **bpy_version** 2.8.2

   ShapeNet_data\/\*\/\*.obj -> \/obj\*\/\*.obj, *mtl and \/ply\*\/\*.ply \ 

   ---
2. **Get rendered points based on info file**
   ```
   ./blensor_1.0.18-RC10_x64/Blensor-x64.AppImage assets/table_cycles_blensor_final.blend --python render_cloud.py
   ```
   ```
   python pcd_transform.py
   ```
   @param: **server = BlensorSceneServer(0,5999)** start and end number
   \/\*.json and \/obj\*\/\*.obj, *mtl -> \/rendered_simple\/\*.pcd

   ---
3. **Get scene points and Get frames of single object**
   ```
   python data_object_darboux_generator_final.py
   ```
   \/ply\/\*.ply -> \/data_scene\/\*.p

   ---
4. **Get frames based on scene points and rendered points** 
   ```
   python generate_fast_training_data.py
   ```
   \/data_scene\/\*.p & \/rendered_simple\/\*.pcd -> \/eval_data\/\*.p

   ---