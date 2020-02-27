import _fix_path
import os
import subprocess
import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image
from os.path import join as pjoin

from traindet import config as cfg

def add_background(rendered_folder, output_folder, args, gray=False):
    
    rendered_folder = Path(rendered_folder)
    backgrounds_folder = Path(cfg.backgrounds_folder)
    output_folder =  Path(output_folder)

    classes = os.listdir(rendered_folder)

    bgs = [backgrounds_folder / fn for fn in os.listdir(backgrounds_folder)]

    random.seed(args.seed)
    for _class in classes:
        cls_path = rendered_folder / _class
        for fn in os.listdir(cls_path):
            fn = Path(fn)
            if fn.suffix == '.png':
                origin_path = cls_path / fn
                print(f'Processing file {origin_path}')
                if gray:
                    bg = pjoin(cfg.assets_folder, 'gray_bg.jpeg')
                else:
                    bg = random.choice(bgs)
                background = Image.open(bg)
                foreground = Image.open(origin_path)
                background = background.resize(foreground.size, Image.ANTIALIAS)
                background.paste(foreground, (0, 0), foreground)
                os.makedirs(output_folder / _class, exist_ok=True)
                background.save(output_folder / _class / fn)
                origin_label = os.path.splitext(str(origin_path))[0] + '.txt'
                shutil.copy(origin_label, output_folder / _class / (str(fn).split('.')[0] + '.txt'))
            
    shutil.rmtree(rendered_folder)


def split_dataset(dataset_folder, args):

    seed = args.seed
    train_fraq = args.train_fraq
    random.seed(seed)
    data = []
    for folder in sorted(os.listdir(dataset_folder)):
        if not os.path.isdir(pjoin(dataset_folder, folder)): continue
        for item in sorted(list(set([fn.split('.')[0] for fn in sorted(os.listdir(pjoin(dataset_folder, folder)))]))):
            entry = {}
            entry['img'] = pjoin(folder, f'{item}.png')
            entry['label'] = pjoin(folder, f'{item}.txt')
            data.append(entry)

    random.shuffle(data)

    train = data[:int(len(data) * train_fraq)]
    test = data[int(len(data) * train_fraq):]

    with open(pjoin(dataset_folder, 'train.json'), 'w') as f:
        json.dump(train, f)

    with open(pjoin(dataset_folder, 'test.json'), 'w') as f:
        json.dump(test, f)


def parse_args():

    parser = argparse.ArgumentParser('Render parts images using blender.')
    parser.add_argument('--dataset-name', help='The name of the output dataset.')
    parser.add_argument('--train-fraq', default=0.7)
    parser.add_argument('--seed', default=233, type=int)
    parser.add_argument('--output-folder', default=pjoin(cfg.project_folder, 'temp'))
    parser.add_argument('--background', action='store_true')
    
    parser.add_argument('--mode', default='test', 
        help='The mode of the rendering, options: random, deterministic, test')
    
    parser.add_argument('--vangles', default='-90,90,30', help='Vertical angle,\
         three values on deterministic mode (main,max,step), two on random mode \
        (min,max)')
    parser.add_argument('--hangles', default='0,90,30', help='Horizontal angle,\
         three values on deterministic mode (main,max,step), two on random mode\
         (min,max)')
    parser.add_argument('--distances', default='300,600,1000', 
        help='Camera distances from object, list of values on deterministic mode \
        , two values on random mode (min,max)')

    parser.add_argument('--nviews', default=3, help='Number of views per object on random mode')

    parser.add_argument('--noise-std', default=0.3, type=float, help='Gaussian noise \
        standard deviation in a fraction of the step in deterministic mode')

    parser.add_argument('--num_lamps', default=3, help='Number of lamps on scene.')
    parser.add_argument('--light_power', default='5,6.5', help='(10**power_min, 10**power_max')
    
    # parser.add_argument('--noise-std', default=0, help='Standard deviation of gaussian noise added to ')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    command = """
        blender
        --background
        --python rendering/entry.py
        --
    """
    # propagate all the original args to blender script
    for key, val in args._get_kwargs():
        command += f' --{key} {val}'

    # add folders to args
    extra_args = {
        'assets_folder': cfg.assets_folder,
        'parts_folder': cfg.parts_folder,
    }
    for key, val in extra_args.items():
        command += f' --{key} {val}'

    subprocess.call(command.split())

    rendered_folder = pjoin(args.output_folder, 'rendered_images')
    dataset_folder = pjoin(args.output_folder, args.dataset_name)
    if args.background:
        add_background(rendered_folder, dataset_folder, args)
    else:
        # os.rename(rendered_folder, dataset_folder)
        add_background(rendered_folder, dataset_folder, args, gray=True)
    
    split_dataset(dataset_folder, args)

if __name__ == '__main__':

    main()