from detection import Detector
from traindet.train_utils import get_dataset
import time
import numpy as np
_, val_ds, _ = get_dataset('real')

det = Detector(model='ssd512_mobile', dataset='real', ctx='gpu')

times = []
for x in val_ds:
    tic = time.time()
    img = x[0].asnumpy()
    bboxes, rimg = det.detect(img)
    times.append(time.time() - tic)

m_time = np.mean(times)
print(m_time)
print(1/m_time)



# import cv2
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from traindet.train_utils import get_dataset
# from traindet.utils import load_predictions
# from traindet.config import classes

# from bboxes.main import BboxList, Bbox

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

# import time
# from bboxes.window import VideoWindow

# from traindet.train_utils import get_dataset
# from detection.window import VideoOutput
# import time
# # import os
# # os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
# import cv2
# _, val_ds, _ = get_dataset('real')
# vo = VideoOutput('Model')
# for x in val_ds:
#     img = x[0].asnumpy().astype('int')
#     vo.send_frame(img)
#     time.sleep(0.5)
# for x in val_ds:
#     img = x[0].asnumpy()/255
#     print(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow('Model', img)
#     cv2.waitKey(1)
#     time.sleep(1)

# cv2.destroyAllWindows()

# import matplotlib.pyplot as plt
# import cv2
# from detection import Detector
# from traindet.train_utils import get_dataset
# from traindet.utils import load_predictions

# _, val_ds, _ = get_dataset('real')
# img = val_ds[0][0].asnumpy()
# img.shape


# _, val_ds, _ = get_dataset('real')
# img = val_ds[0][0].asnumpy().astype('int')
# preds = load_predictions('real')
# pred = preds['ssd300']['preds'][0]
# rimg = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_NEAREST)

# from gluoncv.utils.viz import cv_plot_bbox
# img = cv_plot_bbox(rimg, pred[:, 0:4], scores=pred[:, 5], labels=pred[:, 4], thresh=0.5,
#                  class_names=val_ds.classes, colors=None,
#                  absolute_coordinates=True, scale=1.0)

# plt.imshow(img)
# plt.show()
