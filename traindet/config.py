import os
from os.path import join as pjoin
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

# Folders
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = pjoin(project_folder, 'data')
dataset_folder = pjoin(data_folder, 'datasets')
gen_data_folder = pjoin(data_folder, 'gen_data')
checkpoints_folder = pjoin(data_folder, 'checkpoints')
# checkpoints_folder = pjoin(project_folder, 'checkpoints')
outputs_folder = pjoin(data_folder, 'outputs')

assets_folder = pjoin(data_folder, 'assets')
parts_folder = pjoin(assets_folder, 'stl_models')
backgrounds_folder = pjoin(assets_folder, 'backgrounds')

# Datasets
real_dataset_folder = pjoin(dataset_folder, 'real')
real_grasp_dataset_folder = pjoin(dataset_folder, 'real_with_grasp')
synth_dataset_folder = pjoin(dataset_folder, 'synth')

used_model_names = ['ssd300', 'ssd512', 'yolo416', 'frcnn'] # For compatibility
model_names = ['ssd300', 'ssd512', 'yolo416', 'frcnn']

formated_model_names = ['SSD300', 'SSD512', 'YOLOv3-416', 'Faster R-CNN']

# Names

classes = [
    'tray',
    'dosing_nozzle',
    'button_pad',
    'part1',
    'part2',
    'part3'
    ]

formated_classes = [
    'Tray', 
    'Dosing Nozzle', 
    'Button Pad', 
    'Part 01', 
    'Part 02', 
    'Part 03']


classes_grasp = [
    'tray',
    'dosing_nozzle',
    'button_pad',
    'part1',
    'part2',
    'part3',
    'grasping_tag',
    'grasping_cylinder',
    'grasping_cuboid']



dataset_names = [
    'real',  # Real Object Photos
    'synth_small_nobg',
    'synth_small_bg'
    ]

# Metric
eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)