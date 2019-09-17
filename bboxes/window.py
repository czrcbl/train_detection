import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from traindet.train_utils import get_dataset
from multiprocessing import Process, Queue

matplotlib.use("TkAgg")  
# turn navigation toolbar off
plt.rcParams['toolbar'] = 'None'

# _, val_ds, _ = get_dataset('real')
# nx = 150
# ny = 50

# img = None
# for i in range(10):
#     im = val_ds[i][0].asnumpy().astype('int')
#     if img is None:
#         img = plt.imshow(im)
#         plt.axis('off')
#     else:
#         img.set_data(im)
#     plt.pause(1)
#     plt.draw()


class VideoWindow:

    def __init__(self):
        self.img = None
        self.q = Queue()
        self.process = Process(target=self.process_f)
        self.process.start()
        
    def process_f(self):
        frame = self.q.get()
        if self.img is None:
            self.img = plt.imshow(frame)
            plt.axis('off')
        else:
            self.img.set_data(frame)
        plt.draw()

    def send_frame(self, frame):
        self.q.put(frame)