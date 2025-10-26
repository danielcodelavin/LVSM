import bpy
import os
import sys
import glob
from mathutils import Vector  


INPUT_DIRECTORY = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/reorganized_gso"
OUTPUT_DIRECTORY = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/converted_glb_gso"




def clean_scene():
    """Clears all mesh objects from the scene."""
    if bpy.context.object and bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0: bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0: bpy.data.images.remove(block)


def rebuild_materials(object_dir):
    """
    Deletes all existing materials and creates a new one from scratch,
    linking the object's texture file to the Base Color.
    """
    print("  Rebuilding materials from scratch...")

   
    texture_path = None
    texture_dir = os.path.join(object_dir, 'materials', 'textures')
    if os.path.isdir(texture_dir):
        found_textures = glob.glob(os.path.join(texture_dir, '*.png')) + \
                         glob.glob(os.path.join(texture_dir, '*.jpg')) + \
                         glob.glob(os.path.join(texture_dir, '*.jpeg'))
        if found_textures:
            texture_path = found_textures[0]
            print(f"    - Found texture: {os.path.basename(texture_path)}")
        else:
            print("    - No texture image file found in materials/textures folder.")
            return
    else:
        print("    - No 'materials/textures' directory found.")
        return

    
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue

        print(f"    - Processing object: {obj.name}")
        obj.data.materials.clear()

        mat = bpy.data.materials.new(name="PBR_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")

        if not bsdf:
            print("    - ERROR: Could not find Principled BSDF node.")
            continue

        tex_image_node = nodes.new('ShaderNodeTexImage')
        tex_image_node.location = bsdf.location - Vector((300, 0))

        try:
            tex_image_node.image = bpy.data.images.load(texture_path)
        except Exception as e:
            print(f"    - ERROR: Could not load texture image '{texture_path}': {e}")
            continue

        print("    - Linking new texture node to Base Color.")
        links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
        obj.data.materials.append(mat)


def main():
    """Main execution function."""
    print("--- Starting Final Batch Conversion ---")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    object_folders = sorted([f for f in os.listdir(INPUT_DIRECTORY) if os.path.isdir(os.path.join(INPUT_DIRECTORY, f))])

    for object_name in object_folders:
        print(f"\n--- Processing: {object_name} ---")
        object_dir = os.path.join(INPUT_DIRECTORY, object_name)
        obj_path = os.path.join(object_dir, 'meshes', 'model.obj')

        if not os.path.exists(obj_path):
            print(f"  [!] Skipping: '{obj_path}' not found.")
            continue

        glb_path = os.path.join(OUTPUT_DIRECTORY, f"{object_name}.glb")
        clean_scene()

        try:
            bpy.ops.import_scene.obj(filepath=obj_path)
            rebuild_materials(object_dir)
            bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', export_yup=True)
            print(f"  [SUCCESS] Converted {object_name}")
        except Exception as e:
            print(f"  [FAILURE] Could not convert {object_name}: {e}")

    print("\n--- Final Conversion Finished ---")

if __name__ == "__main__":
    import addon_utils
    addon_utils.enable("io_scene_obj", default_set=True)
    addon_utils.enable("io_scene_gltf2", default_set=True)
    main()