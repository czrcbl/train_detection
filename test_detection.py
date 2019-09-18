from detection.main import Detector
from detection import config as cfg
from os.path import join as pjoin
from matplotlib.image import imread
import matplotlib.pyplot as plt

img = imread(pjoin(cfg.dataset_folder, 'real/test/a0rc.png'))
det = Detector()
bboxes, _ = det.detect(img, threshold=0.5)
dimg = bboxes.draw(img)
plt.imshow(dimg)
plt.show()
