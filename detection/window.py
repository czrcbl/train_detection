import numpy as np
import matplotlib
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation
from multiprocessing import Process, Queue

# matplotlib.use("TkAgg")  
# # turn navigation toolbar off
# plt.rcParams['toolbar'] = 'None'

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


class VideoOutput:

    def __init__(self, window_name):
        self.window_name = window_name
        self.img = None
        self.q = Queue()
        self.process = Process(target=self.process_f)
        self.process.start()
        
    def process_f(self):
        while True:
            frame = self.q.get()
            cv2.imshow(self.window_name, frame)

    def send_frame(self, frame):
        self.q.put(frame)