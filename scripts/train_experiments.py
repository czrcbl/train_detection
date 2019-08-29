import subprocess
import argparse
import os


def train_ssd300(epochs):

    if epochs is None:
        epochs = 100

    command = f"""
        python scripts/train_ssd.py 
            --transfer
            --base-model ssd_300_vgg16_atrous_coco
            --data-shape 300
            --dataset real
            --save-prefix checkpoints/ssd300/
            --batch-size 4
            --epochs {epochs}
            --lr 0.001
            --lr-decay 0.1
            --lr-decay-epoch 60,80
            --seed 233
    """

    subprocess.call(command.split())


def train_ssd512(epochs):
    if epochs is None:
        epochs = 100

    command = f"""
        python scripts/train_ssd.py 
            --transfer
            --base-model ssd_512_resnet50_v1_coco
            --data-shape 512
            --dataset real
            --save-prefix checkpoints/ssd512/
            --batch-size 4
            --epochs {epochs}
            --lr 0.001
            --lr-decay 0.1
            --lr-decay-epoch 60,80
            --seed 233
    """

    subprocess.call(command.split())


def train_frcnn(epochs):
    
    if epochs is None:
        epochs = 50

    command = f"""
        python scripts/train_faster_rcnn.py
            --transfer
            --base-model faster_rcnn_resnet50_v1b_coco
            --dataset real
            --save-prefix checkpoints/faster_rcnn/
            --epochs {epochs}
            --lr 0.001
            --lr-decay 0.1
            --lr-decay-epoch 30,40
            --seed 233
    """

    subprocess.call(command.split())


def train_yolo3(epochs):
    
    if epochs is None:
        epochs = 100

    command = f"""
        python scripts/train_yolo3.py
            --transfer
            --base-model yolo3_darknet53_coco
            --data-shape 416
            --dataset real
            --batch-size 4
            --save-prefix checkpoints/yolo416/
            --epochs {epochs}
            --lr 0.0001
            --lr-decay 0.1
            --lr-decay-epoch 60,80
            --seed 233
            --no-random-shape
    """

    subprocess.call(command.split())


def main():

    parser = argparse.ArgumentParser(description='Train Networks')
    parser.add_argument('--test', action='store_true',
                        help="Whether it is a test.")
    parser.add_argument('--model', type=str, default='',
                        help="Model to train.")

    args = parser.parse_args()

    if args.test:
        epochs = 1
    else:
        epochs = None

    if args.model == '':
        raise ValueError('You should provide a model.')
    elif args.model == 'ssd300':
        train_ssd300(epochs)
    elif args.model == 'ssd512':
        train_ssd512(epochs)
    elif args.model == 'frcnn':
        train_frcnn(epochs)
    elif args.model == 'yolo': 
        train_yolo3(epochs)
    elif args.model == 'all':
        train_ssd300(epochs)
        train_ssd512(epochs)
        train_yolo3(epochs)
        train_frcnn(epochs)
    else:
        raise ValueError('Invalid argument for model.')

    
if __name__ == '__main__':
    main()
