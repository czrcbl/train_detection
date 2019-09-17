import _fix_path
from traindet import config as cfg
import subprocess


def train_ssd512_mobile(epochs, dataset):
    if epochs is None:
        epochs = 100

    command = f"""
        python scripts/train_ssd.py 
            --transfer
            --base-model ssd_512_mobilenet1.0_coco
            --data-shape 512
            --dataset {dataset}
            --save-prefix {cfg.checkpoints_folder}/{dataset}/ssd512_mobile/
            --batch-size 4
            --epochs {epochs}
            --lr 0.001
            --lr-decay 0.1
            --lr-decay-epoch 60,80
            --seed 233
    """

    subprocess.call(command.split())


def train_ssd512_test(epochs, dataset):
    if epochs is None:
        epochs = 100

    command = f"""
        python scripts/train_ssd.py 
            --transfer
            --base-model ssd_512_resnet50_v1_coco
            --data-shape 512
            --dataset {dataset}
            --save-prefix {cfg.checkpoints_folder}/{dataset}/ssd512_test/
            --batch-size 4
            --epochs {epochs}
            --lr 0.001
            --lr-decay 0.1
            --lr-decay-epoch 60,80
            --seed 233
    """

    subprocess.call(command.split())


if __name__ == '__main__':

    train_ssd512_mobile(None, 'real')
    # train_ssd512_test(None, 'real')