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

def add_background(args):
    
    rendered_folder = Path(f'{cfg.assets_folder}/rendered_images/{args.mode}')
    backgrouds_folder = Path(cfg.backgrounds_folder)
    output_folder = Path(cfg.dataset_folder) / Path(args.out_dataset_folder)

    classes = os.listdir(rendered_folder)

    bgs = [backgrouds_folder / fn for fn in os.listdir(backgrouds_folder)]

    random.seed(args.seed)
    for _class in classes:
        cls_path = rendered_folder / _class
        for fn in os.listdir(cls_path):
            fn = Path(fn)
            if fn.suffix == '.png':
                origin_path = cls_path / fn
                print(f'Processing file {origin_path}')
                bg = random.choice(bgs)
                background = Image.open(bg)
                foreground = Image.open(origin_path)
                background = background.resize(foreground.size, Image.ANTIALIAS)
                background.paste(foreground, (0, 0), foreground)
                os.makedirs(output_folder / _class, exist_ok=True)
                background.save(output_folder / _class / fn)
                origin_label = os.path.splitext(str(origin_path))[0] + '.txt'
                shutil.copy(origin_label, output_folder / _class / (str(fn).split('.')[0] + '.txt'))
            

def split_dataset(args):

    seed = args.seed
    train_fraq = args.train_fraq
    dataset_folder = pjoin(cfg.dataset_folder, args.out_dataset_folder)
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

    parser = argparse.ArgumentParser('Render parts images on blender.')
    parser.add_argument('--mode', default='test', 
        help='The mode of the rendering.')
    parser.add_argument('--out-dataset-folder', default='synth', 
        help='The name of the output dataset.')
    parser.add_argument('--seed', default=233)
    parser.add_argument('--train_fraq', default=0.7)
    args = parser.parse_args()

    assert (args.out_dataset_folder not in os.listdir(cfg.dataset_folder)),\
        'Dataset Output Folder already exists.'
    
    return args


def main():
    args = parse_args()
    command = """
        blender
        --background
        --python rendering/entry.py
        --
    """

    for key, val in args._get_kwargs():
        command += f' --{key} {val}'

    extra_args = {
        'assets_folder': cfg.assets_folder,
        'parts_folder': cfg.parts_folder
    }
    for key, val in extra_args.items():
        command += f' --{key} {val}'

    print(command.split())
    subprocess.call(command.split())

    add_background(args)
    split_dataset(args)

if __name__ == '__main__':

    main()