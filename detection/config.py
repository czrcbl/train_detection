import os
from os.path import join as pjoin
project_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
data_folder = pjoin(project_path, 'data')
dataset_folder = pjoin(data_folder, 'datasets')
checkpoints_folder = pjoin(data_folder, 'checkpoints')
# classes = [
#     'tray',
#     'dosing_nozzle',
#     'button_pad',
#     'part1',
#     'part3',
#     'part2'
#     ]

if __name__ == '__main__':
    pass