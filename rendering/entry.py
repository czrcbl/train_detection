import os
import sys
import argparse
import bpy
import math
from mathutils import Vector
import mathutils
from os.path import join as pjoin
import random

from rendering.render_utils import Box, camera_view_bounds_2d, get_params
from rendering.render_config import parts
from rendering import render_config as rcfg


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


def parse_args():
    
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    args = ArgStore(argv)

    return args


def main():

    args = parse_args()
    # mode = 'small'

    root_dir = args.assets_folder
    output_folder = pjoin(root_dir, f'rendered_images/{args.mode}')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    random.seed(args.seed)

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

    for obj_number, part in enumerate(parts):
        
        clear()
        # scene = bpy.context.scene
        # scene.cycles.device = 'GPU'
        #load and position mesh
        name = part['name']
        obj = load_stl(pjoin(args.parts_folder, f'{name}.stl'))
        center_rotate_obj(obj, part['rot'])

        cam = bpy.data.cameras.new("Camera")
        cam_ob = bpy.data.objects.new("Camera", cam)
        bpy.context.collection.objects.link(cam_ob)
        scene.camera = cam_ob
        
        scene.camera.data.clip_end = 10000

        #write_bounds_2d(filepath, scene, cam_ob, obj, frame_start, frame_end)
        distances, vert_angles, rot_angles = get_params(part, mode=args.mode)
        
        #distances = part['dists']
        #if is_test:
        #    vert_angles = (0, math.pi/6, math.pi/3)[:1]
        #    rot_angles = [2*math.pi/8 * x for x in range(8)][:1]
        #else:
        #    vert_angles = (0, math.pi/6, math.pi/3)
        #    rot_angles = [2*math.pi/8 * x for x in range(8)]

        prefix = pjoin(output_folder, part['name'])
        i = 0
        n = 0
        for obj_rot in (0, math.pi/2):
            obj.rotation_euler = tuple([a + b for a, b in zip(part['rot'], (obj_rot, 0, 0))]) 
            for dist in distances:
                n += 1
                for vert_angle in vert_angles:
                    for rot_angle in rot_angles:
                        
                        z = dist * math.sin(vert_angle)
                        d = dist * math.cos(vert_angle)
                        x = d * math.cos(rot_angle)
                        y = d * math.sin(rot_angle)
                        
                        l_z = 10 * math.sin(vert_angle)
                        l_d = 10 * math.cos(vert_angle)
                        l_x = l_d * math.cos(rot_angle)
                        l_y = l_d * math.sin(rot_angle)
                        
                        cam_ob.location = (x, y, z)
                        pos = update_camera(cam_ob, Vector((0, 0, 0)))
                        del lamp
                        lamp = create_lamp(Vector((0, 0, 0)), energy)
                        lamp.location = pos + Vector((l_x, l_y, l_z))
                        fprefix = pjoin(prefix, str(i))
                        
                        render_scene_bb(scene, cam_ob, obj, obj_number, fprefix)
                        
                        i += 1

if __name__ == '__main__':
    
    main()

