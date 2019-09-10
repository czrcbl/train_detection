import os
from os.path import join as pjoin
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

# Folders
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dataset_folder = pjoin(project_folder, 'datasets')
gen_data_folder = pjoin(project_folder, 'gen_data')

project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
assets_folder = pjoin(project_folder, 'assets')
parts_folder = pjoin(assets_folder, 'stl_models')
backgrounds_folder = pjoin(assets_folder, 'backgrounds')

# Datasets
real_dataset_folder = pjoin(dataset_folder, 'real')
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


dataset_names = ['real', 'synth', 'mixed']

# Metric
eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)