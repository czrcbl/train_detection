import os
from os.path import join as pjoin
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dataset_folder = pjoin(project_folder, 'datasets')
real_dataset_folder = pjoin(dataset_folder, 'real')

used_model_names = ['ssd300', 'ssd512', 'yolo416', 'frcnn']

formated_model_names = ['SSD300', 'SSD512', 'YOLOv3-416', 'Faster R-CNN']

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

eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)