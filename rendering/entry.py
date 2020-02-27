import sys
import os
path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)
import argparse
import bpy
import math
from mathutils import Vector
import mathutils
from os.path import join as pjoin
import random

print(sys.path)
# from rendering.render_utils import Box, camera_view_bounds_2d, get_params
# from rendering.render_config import parts
# from rendering import render_config as rcfg

# from .render_utils import Box, camera_view_bounds_2d, get_params
# from .render_config import parts
# from . import render_config as rcfg

from render_utils import Box, camera_view_bounds_2d, get_params
from render_config import parts
import render_config as rcfg


class ArgStore:

    def __init__(self, args):
        assert len(args)%2 == 0
        for i in range(0, len(args), 2):
            self.__dict__[args[i].strip('--')] = args[i+1]
    
    def __repr__(self):
        return repr(self.__dict__)


def get_center(o):
    local_bbox_center = 0.125 * sum((Vector(b) for b in o.bound_box), Vector())
    global_bbox_center = o.matrix_world @ local_bbox_center
    
    return global_bbox_center

def clear():
    # Select objects by type
    for o in bpy.context.scene.objects:
        o.select_set(True)
    # Call the operator
    bpy.ops.object.delete()
    

def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=10.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    new_position = camera.location + rot_quat @ mathutils.Vector((0.0, 0.0, distance))
    camera.location = new_position
    
    return new_position


def create_lamp(point, energy=1000):
    lamp_data = bpy.data.lights.new(name="Lamp", type='POINT')
    lamp_data.energy = energy
    lamp_object = bpy.data.objects.new(name="Lamp", object_data=lamp_data)
    bpy.context.collection.objects.link(lamp_object)
    bpy.context.view_layer.objects.active = lamp_object
    lamp_object.location = point
    
    return lamp_object

def load_stl(filepath):
    bpy.ops.import_mesh.stl(filepath=filepath)
    obj = bpy.context.object
    
    return obj

def center_rotate_obj(obj, angles):
    bpy.context.scene.cursor.location = get_center(obj)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    obj.location=(0,0,0)
    obj.rotation_euler = angles
    


def custom2yolo(data):
    """Convert yolo format to voc, both in relative coordinates."""
    yolo = []
    bbox_width = float(data[3]) 
    bbox_height = float(data[4])
    x0 = float(data[1])
    y0 = float(data[2])
    yolo.append(data[0])
    yolo.append(x0 + (bbox_width / 2))
    yolo.append(y0 + (bbox_height / 2))
    yolo.append(bbox_width)
    yolo.append(bbox_height)
    
    return yolo

def convertbb(bb):
    return bb[0] / 800.0, bb[1]/600.0, bb[2]/800.0, bb[3]/600.0

def render_scene_bb(scene, cam_ob, obj, obj_number, prefix):
    
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.image_settings.file_format='PNG'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.data.scenes['Scene'].render.filepath = prefix + '.png'
    bpy.ops.render.render(write_still=True)
    
    
    filepath = prefix + '.txt'
    coords = camera_view_bounds_2d(scene, cam_ob, obj).to_tuple()
    coords = list(convertbb(coords))
    coords.insert(0, obj_number)
    coords = custom2yolo(coords)
    with open(filepath, "w") as file:
        file.write("%i %f %f %f %f\n" % tuple(coords))


def listdir(pdir):
    return [pjoin(pdir, x) for x in sorted(os.listdir(pdir))]


def add_texture(texture_path, obj):
    mat = bpy.data.materials.new(name='texture')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    # bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(texture_path)
    # mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    # disp = nodes['Material Output'].inputs['Displacement']
    # mat.node_tree.links.new(disp, texImage.outputs['Color'])

    principled = nodes['Principled BSDF']

    mat.node_tree.links.new(texImage.outputs[0], principled.inputs[0])
    # Assign it to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# def add_texture(img_path, obj):

#     material_obj = bpy.data.materials.new('number_1_material')

#     image_obj = bpy.data.images.load(img_path)

#     texture_obj = bpy.data.textures.new('number_1_tex', type='IMAGE')
#     texture_obj.image = image_obj

#     texture_slot = material_obj.texture_slots.add()
#     texture_slot.texture = texture_obj

#     bpy.context.object.data.materials.append(material_obj)

# def add_texture(texture_path, obj):
#     mat = bpy.data.materials.new(name='texture')
#     mat.use_nodes = True
#     bsdf = mat.node_tree.nodes["Principled BSDF"]
#     texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
#     texImage.image = bpy.data.images.load(texture_path)
#     mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

#     # Assign it to object
#     if obj.data.materials:
#         obj.data.materials[0] = mat
#     else:
#         obj.data.materials.append(mat)


# def add_texture(texture_path, obj):
#     mat = bpy.data.materials.new(name='texture')
#     mat.use_nodes = True

#     nodes = mat.node_tree.nodes
#     links = mat.node_tree.links
#     bsdf = nodes["Principled BSDF"]

#     output  = nodes.new("ShaderNodeOutputMaterial")
#     diffuse = nodes.new("ShaderNodeBsdfDiffuse")
#     texture = nodes.new("ShaderNodeTexImage")
#     uvmap   = nodes.new("ShaderNodeUVMap")

#     # texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
#     texture.image = bpy.data.images.load(texture_path)

#     # mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
#     links.new(output.inputs['Surface'], diffuse.outputs['BSDF'])
#     links.new(diffuse.inputs['Color'],   texture.outputs['Color'])
#     links.new(texture.inputs['Vector'],    uvmap.outputs['UV'])
    
#     # Assign it to object
#     if obj.data.materials:
#         obj.data.materials[0] = mat
#     else:
#         obj.data.materials.append(mat)

# def add_texture(texture_path, obj):

#     mat_name = "MyMaterial"
#     image_path = #...

#     mat = (bpy.data.materials.get(mat_name) or
#         bpy.data.materials.new(mat_name))

#     mat.use_nodes = True
#     nt = mat.node_tree
#     nodes = nt.nodes
#     links = nt.links

#     # clear
#     while(nodes): nodes.remove(nodes[0])

#     output  = nodes.new("ShaderNodeOutputMaterial")
#     diffuse = nodes.new("ShaderNodeBsdfDiffuse")
#     texture = nodes.new("ShaderNodeTexImage")
#     uvmap   = nodes.new("ShaderNodeUVMap")

#     texture.image = bpy.data.images.load(image_path)
#     uvmap.uv_map = "UV"

#     links.new( output.inputs['Surface'], diffuse.outputs['BSDF'])
#     links.new(diffuse.inputs['Color'],   texture.outputs['Color'])
#     links.new(texture.inputs['Vector'],    uvmap.outputs['UV'])

#     # distribute nodes along the x axis
#     for index, node in enumerate((uvmap, texture, diffuse, output)):
#         node.location.x = 200.0 * index


#     ColoredGround = create_cycles_material('GroundCol')
#     setMaterial(g_o, ColoredGround)

# def create_cycles_material(name):

#     mat = bpy.data.materials.new(name)
#     mat.use_nodes = True
#     nodes = mat.node_tree.nodes

#     node = nodes['Diffuse BSDF']
#     node.location = 600, 120
#     #pass   # Some more node-engineering here
#     return mat

# def setMaterial(ob, mat):
#     me = ob.data
#     me.materials.append(mat)

def create_parameter_range(arg):
    vals = [float(a) for a in arg.strip('"').split(',')]
    out = []
    v = vals[0]
    while v <= vals[1]:
        out.append(v)
        v += vals[-1]
    if out[-1] < vals[1]:
        out.append(vals[1])
    
    return out, vals[-1]

def deterministic_render(args):

    random.seed(int(args.seed))

    noise_std = float(args.noise_std)

    root_dir = args.output_folder
    output_folder = pjoin(root_dir, f'rendered_images')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    energy = 10**6
    lamp = None


    prefs = bpy.context.preferences
    cuda_devices, opencl_devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # Set GPU rendering
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = 'CUDA'
    except TypeError:
        print("Could not enable CUDA")
        raise ValueError('GPU not supported')

    for device in cuda_devices:
        print(f'Activating {device.name}')
        device.use = True

    rot_angles, rot_step = create_parameter_range(args.hangles)
    vert_angles, vert_step = create_parameter_range(args.vangles)
    distances = [float(a) for a in args.distances.split(',')]

    for obj_number, part in enumerate(parts):
        
        clear()
        name = part['name']
        obj = load_stl(pjoin(args.parts_folder, f'{name}.stl'))
        center_rotate_obj(obj, part['rot'])

        cam = bpy.data.cameras.new("Camera")
        cam_ob = bpy.data.objects.new("Camera", cam)
        bpy.context.collection.objects.link(cam_ob)
        scene.camera = cam_ob
        
        scene.camera.data.clip_end = 10000

        prefix = pjoin(output_folder, part['name'])
        i = 0
        n = 0
        # for obj_rot in (0, math.pi/2):
        # obj.rotation_euler = tuple([a + b for a, b in zip(part['rot'], (obj_rot, 0, 0))]) 
        obj.rotation_euler = part['rot']
        for dist in distances:
            n += 1
            for vert_angle in vert_angles:
                for rot_angle in rot_angles:
                    distr = random.gauss(dist, noise_std * dist)
                    vert_angler = random.gauss(vert_angle, noise_std * vert_step)
                    rot_angler = random.gauss(rot_angle, noise_std * rot_step)

                    z = distr * math.sin(vert_angler)
                    d = distr * math.cos(vert_angler)
                    x = d * math.cos(rot_angler)
                    y = d * math.sin(rot_angler)
                    
                    l_z = 10 * math.sin(vert_angler)
                    l_d = 10 * math.cos(vert_angler)
                    l_x = l_d * math.cos(rot_angler)
                    l_y = l_d * math.sin(rot_angler)
                    
                    cam_ob.location = (x, y, z)
                    pos = update_camera(cam_ob, Vector((0, 0, 0)))
                    del lamp
                    lamp = create_lamp(Vector((0, 0, 0)), energy)
                    lamp.location = pos + Vector((l_x, l_y, l_z))
                    fprefix = pjoin(prefix, str(i))
                    
                    render_scene_bb(scene, cam_ob, obj, obj_number, fprefix)
                    
                    i += 1


def random_render(args):
    """Render the object in random positions."""

    random.seed(args.seed)

    root_dir = args.output_folder
    output_folder = pjoin(root_dir, f'rendered_images')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    textures = listdir(pjoin(args.assets_folder, 'textures'))
    random.seed(args.seed)
    distances = create_parameter_range(args.distances)
    # n_views = 10
    params_range = {
        'energy': (5*10e5, 10e6),
        'rot_x': (0, 2 * math.pi),
        'rot_y': (0, 2 * math.pi),
        'rot_z': (0, 2 * math.pi),
        'distance': distances,
        'lamp_r': (300, 1000)
    }

##########################################

    prefs = bpy.context.preferences
    cuda_devices, opencl_devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # Set GPU rendering
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = 'CUDA'
    except TypeError:
        print("Could not enable CUDA")
        raise ValueError('GPU not supported')

    for device in cuda_devices:
        print(f'Activating {device.name}')
        device.use = True

 ##########################################

    for obj_number, part in enumerate(parts):
        
        clear()
        name = part['name']
        obj = load_stl(pjoin(args.parts_folder, f'{name}.stl'))
        cam = bpy.data.cameras.new("Camera")
        cam_ob = bpy.data.objects.new("Camera", cam)
        bpy.context.collection.objects.link(cam_ob)
        scene.camera = cam_ob
        
        scene.camera.data.clip_end = 10000

        prefix = pjoin(output_folder, part['name'])

        for i in range(int(args.nviews)):
            texture = random.choice(textures)
            add_texture(texture, obj)
            center_rotate_obj(obj, part['rot'])
            rot_x = random.uniform(*params_range['rot_x'])
            rot_y = random.uniform(*params_range['rot_y'])
            rot_z = random.uniform(*params_range['rot_z'])
            dist = random.uniform(*params_range['distance'])
            energy = random.uniform(*params_range['energy'])

            obj.rotation_euler = (rot_x, rot_y, rot_z)
            # cam_ob.location = (dist, y, z)
            cam_ob.location = Vector((dist, 0, 0))
            pos = update_camera(cam_ob, Vector((0, 0, 0)))
            lx = random.uniform(*params_range['lamp_r'])
            ly = random.uniform(*params_range['lamp_r'])
            lz = random.uniform(*params_range['lamp_r'])
            lamp = create_lamp(Vector((lx, ly, lz)), energy)

            fprefix = pjoin(prefix, str(i))
            render_scene_bb(scene, cam_ob, obj, obj_number, fprefix)


def parse_args():
    
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    args = ArgStore(argv)

    return args

    
def main():

    args = parse_args()

    if args.mode == 'random':
        random_render(args)
    elif args.mode == 'deterministic':
        deterministic_render(args)
    elif args.mode == 'random':
        random_render(args)


if __name__ == '__main__':
    
    main()

