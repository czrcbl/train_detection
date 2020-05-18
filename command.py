"""Helper file to call scripts and register inputs on code."""
import subprocess
import argparse
from easydict import EasyDict as edict
from traindet import config as cfg

epochs = 50

dataset_creation = []
model_training = []

cmds = edict()

cmds.synth_test = f"""
    python scripts/render.py
        --mode deterministic
        --dataset-name synth_test
        --hangles 0,90,90
        --vangles "-90,90,90"
        --distances 300,600,900,1200
        --background
        --noise-std 0.3
"""

cmds.synth_small_nobg = f"""
    python scripts/render.py
        --mode deterministic
        --dataset-name synth_small_nobg
        --hangles 0,90,30
        --vangles "-90,90,30"
        --distances 300,600,900
        --noise-std 0.3
"""

# dataset_creation.append(synth_small_nobg)

cmds.synth_small_bg = f"""
        python scripts/render.py
        --mode deterministic
        --dataset-name synth_small_bg
        --hangles 0,90,30
        --vangles "-90,90,30"
        --distances 300,600,900
        --background
        --noise-std 0.3
"""

cmds.synth_medium_bg = f"""
        python scripts/render.py
        --mode deterministic
        --dataset-name synth_medium_bg
        --hangles 0,90,10
        --vangles "-90,90,10"
        --distances 300,600,900,1200
        --background
        --noise-std 0.3
"""

# dataset_creation.append(synth_small_bg)

def default_ssd(name, dataset, epochs):
    ssd = f"""
    python scripts/train_ssd.py 
        --transfer
        --base-model ssd_512_resnet50_v1_coco
        --data-shape 512
        --dataset {dataset}
        --save-prefix {cfg.checkpoints_folder}/{dataset}/{name}/
        --batch-size 4
        --epochs {epochs}
        --lr 0.001
        --lr-decay 0.1
        --lr-decay-epoch 20,40
        --seed 233
    """
    return ssd

def default_frcnn(name, dataset, epochs):
    frcnn = f"""
        python scripts/train_faster_rcnn.py
            --transfer
            --base-model faster_rcnn_resnet50_v1b_coco
            --dataset {dataset}
            --save-prefix {cfg.checkpoints_folder}/{dataset}/{name}/
            --epochs {epochs}
            --lr 0.001
            --lr-decay 0.1
            --lr-decay-epoch 20,40
            --seed 233
    """

    return frcnn

def default_yolo(name, dataset, epochs):
    yolo = f"""
        python scripts/train_yolo3.py
            --transfer
            --base-model yolo3_darknet53_coco
            --data-shape 608
            --dataset {dataset}
            --batch-size 4
            --save-prefix {cfg.checkpoints_folder}/{dataset}/{name}/
            --epochs {epochs}
            --lr 0.0001
            --lr-decay 0.1
            --lr-decay-epoch 20,40
            --seed 233
            --no-random-shape
    """

    return yolo

dataset = 'real'

cmds.ssd_real = default_ssd('ssd_default', dataset, epochs)
cmds.frcnn_real = default_frcnn('frcnn_default', dataset, epochs)
cmds.yolo_real = default_yolo('yolo_default', dataset, epochs)

# model_training.extend([ssd_real, frcnn_real, yolo_real])


dataset = 'synth_small_nobg'

cmds.ssd_small_nobg = default_ssd('ssd_default', dataset, epochs)
cmds.frcnn_small_nobg = default_frcnn('frcnn_default', dataset, epochs)
cmds.yolo_small_nobg = default_yolo('yolo_default', dataset, epochs)

# model_training.extend([ssd_small_nobg, frcnn_small_nobg, yolo_small_nobg])


dataset = 'synth_small_bg'

cmds.ssd_small_bg = default_ssd('ssd_default', dataset, epochs)
cmds.frcnn_small_bg = default_frcnn('frcnn_default', dataset, epochs)
cmds.yolo_small_bg = default_yolo('yolo_default', dataset, epochs)

# model_training.extend([ssd_small_bg, frcnn_small_bg, yolo_small_bg])

dataset = 'synth_spec'
tclass = 'part3'
name = 'frcnn_spec_part3'
frcnn = f"""
    python scripts/train_faster_rcnn.py
        --transfer
        --base-model faster_rcnn_resnet50_v1b_coco
        --dataset {dataset}
        --save-prefix {cfg.checkpoints_folder}/{dataset}/{name}/
        --epochs {epochs}
        --tclass {tclass}
        --lr 0.001
        --lr-decay 0.1
        --lr-decay-epoch 20,40
        --seed 233
"""
cmds.frcnn_spec_part3 = frcnn


dataset = 'real'
name = 'frcnn_test'
frcnn = f"""
    python scripts/train_faster_rcnn.py
        --transfer
        --base-model faster_rcnn_resnet50_v1b_coco
        --dataset {dataset}
        --save-prefix {cfg.checkpoints_folder}/{dataset}/{name}/
        --epochs {epochs}
        --lr 0.001
        --lr-decay 0.1
        --lr-decay-epoch 20,40
        --seed 233
"""
cmds.frcnn_test = frcnn


all_commandas = dataset_creation + model_training


def main():
    
    parser = argparse.ArgumentParser('Execute script with preset options.')
    parser.add_argument('--name', help='Script + arguments name(s) (comma separated).')

    args = parser.parse_args()

    for c in args.name.split(','):
        if c not in cmds:
            raise ValueError(f'Cmd {c} not avaliable, avaliable commands: {cmds.keys()}')
    
    for c in args.name.split(','):
        subprocess.call(cmds[c].split())
    
    


if __name__ == '__main__':

    main()
    # command = frcnn_real
    # subprocess.call(command.split())
    # for cmd in dataset_creation:
        # subprocess.call(cmd.split())

    # for cmd in model_training:
        # subprocess.call(cmd.split())