import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from traindet.train_utils import get_dataset
from multiprocessing import queue, process

matplotlib.use("TkAgg")  
# turn navigation toolbar off
plt.rcParams['toolbar'] = 'None'

_, val_ds, _ = get_dataset('real')
nx = 150
ny = 50

img = None
for i in range(10):
    im = val_ds[i][0].asnumpy().astype('int')
    if img is None:
        img = plt.imshow(im)
        plt.axis('off')
    else:
        img.set_data(im)
    plt.pause(1)
    plt.draw()


class VideoWindow:

    def __init__(self):
        pass