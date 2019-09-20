import os
from detection.main import Detector
from detection import config as cfg
from os.path import join as pjoin
from matplotlib.image import imread
import matplotlib.pyplot as plt


img = imread(pjoin(cfg.dataset_folder, 'real/test/a0rc.png'))
img = (255*img).astype('uint8')
# dataset
# img = imread(pjoin(cfg.dataset_folder, 'real_with_grasp/imgs/0du6.png'))
# det = Detector(model='ssd512', dataset='real_with_grasp')
det = Detector(model='ssd512', dataset='real')
bboxes, _ = det.detect(img)
dimg = bboxes.draw(img)
plt.figure()
plt.imshow(dimg)
plt.figure()
# plt.imshow(bboxes.img.astype('uint8'))
plt.imshow(img)

mimg = bboxes[0].mask_image(img)
plt.figure()
plt.imshow(mimg.astype('uint8'))
plt.show()

# det = Detector(model='ssd512', dataset='real_with_grasp')
# examples_dir = pjoin(cfg.data_folder, 'assets/printer_images')
# file_paths = [pjoin(examples_dir, x) for x in sorted(os.listdir(examples_dir))]
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
# for img_path, ax in zip(file_paths, axs.flat):
#     img = imread(img_path)
#     bboxes, _ = det.detect(img, threshold=0.5)
#     dimg = bboxes.draw(img)
#     ax.set_axis_off()
#     ax.imshow(dimg)

# plt.show()


# det = Detector(model='ssd512', dataset='real_with_grasp')
# examples_dir = pjoin(cfg.data_folder, 'assets/printer_images')
# file_paths = [pjoin(examples_dir, x) for x in sorted(os.listdir(examples_dir))]
# for img_path in file_paths:
#     img = imread(img_path)
#     bboxes, _ = det.detect(img, threshold=0.5)
#     dimg = bboxes.draw(img)
#     plt.figure()
#     plt.imshow(dimg)

# plt.show()
