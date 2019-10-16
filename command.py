import subprocess
from traindet import config as cfg

dataset = 'synth500'
epochs = '40'
    
command = f"""
    python scripts/train_ssd.py 
        --transfer
        --base-model ssd_512_resnet50_v1_coco
        --data-shape 512
        --dataset {dataset}
        --save-prefix {cfg.checkpoints_folder}/{dataset}/ssd512/
        --batch-size 4
        --epochs {epochs}
        --lr 0.001
        --lr-decay 0.1
        --lr-decay-epoch 60,80
        --seed 233
"""

subprocess.call(command.split())