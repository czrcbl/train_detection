import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from traindet.train_utils import get_dataset
from traindet.utils import load_predictions
from traindet.config import classes

from bboxes.main import BboxList, Bbox

# _, val_ds, _ = get_dataset('real')
# img = val_ds[0][0].asnumpy().astype('int')
# preds = load_predictions('real')
# pred = preds['ssd300']['preds'][0]
# rimg = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_NEAREST)
# print(rimg)
# # cv2.imshow('Test image',rimg)

# # rimg = cv2.rectangle(rimg, tuple(pred[0, 0:2]), tuple(pred[0, 2:4]), 1)
# # cv2.putText(outlined_image, 'Fedex', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
# # rimg = cv2.putText(rimg, classes[pred[0, 4]],)
# bbox_list = BboxList(ids=pred[:, 4], scores=pred[:, 5], bboxes=pred[:, :4], th=0.5, img=rimg)
# print(len(bbox_list))
# # bbox = bbox_list[0]
# rimg = bbox_list.draw(rimg)
# plt.imshow(rimg)
# plt.show()


# pred = preds['ssd512']['preds'][0]
# rimg = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
# print(rimg)
# # cv2.imshow('Test image',rimg)

# # rimg = cv2.rectangle(rimg, tuple(pred[0, 0:2]), tuple(pred[0, 2:4]), 1)
# # cv2.putText(outlined_image, 'Fedex', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
# # rimg = cv2.putText(rimg, classes[pred[0, 4]],)
# bbox_list = BboxList(ids=pred[:, 4], scores=pred[:, 5], bboxes=pred[:, :4], th=0.5, img=rimg)
# print(len(bbox_list))
# # bbox = bbox_list[0]
# rimg = bbox_list.draw(rimg)
# plt.imshow(rimg)
# plt.show()

# pred = preds['frcnn']['preds'][0]
# rimg = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_NEAREST)
# print(rimg)
# # cv2.imshow('Test image',rimg)

# # rimg = cv2.rectangle(rimg, tuple(pred[0, 0:2]), tuple(pred[0, 2:4]), 1)
# # cv2.putText(outlined_image, 'Fedex', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
# # rimg = cv2.putText(rimg, classes[pred[0, 4]],)
# bbox_list = BboxList(ids=pred[:, 4], scores=pred[:, 5], bboxes=pred[:, :4], th=0.5, img=rimg)
# print(len(bbox_list))
# # bbox = bbox_list[0]
# rimg = bbox_list.draw(rimg)
# plt.imshow(rimg)
# plt.show()

import time
from bboxes.window import VideoWindow

_, val_ds, _ = get_dataset('real')
vw = VideoWindow()
for x in val_ds:
    img = x[0].asnumpy().astype('int')
    vw.send_frame(img)
    time.sleep(0.5)