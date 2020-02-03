"""Helper file to call scripts and register inputs on code."""
import subprocess
from traindet import config as cfg


# dataset = 'mixed'
# epochs = '100'


# command = f"""
#     python scripts/train_ssd.py 
#         --transfer
#         --base-model ssd_512_resnet50_v1_coco
#         --data-shape 512
#         --dataset {dataset}
#         --save-prefix {cfg.checkpoints_folder}/{dataset}/ssd512_mixed_time_benchmark/
#         --batch-size 4
#         --epochs {epochs}
#         --lr 0.001
#         --lr-decay 0.1
#         --lr-decay-epoch 60,80
#         --seed 233
# """


if __name__ == '__main__':
    subprocess.call(command.split())