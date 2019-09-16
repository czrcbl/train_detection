import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from traindet.train_utils import get_dataset
from traindet.utils import load_predictions
from traindet.config import classes


CMAP = matplotlib.cm.get_cmap('Spectral')


def center_width_height2xyxy(data):
    out = np.zeros((4,))
    bbox_width = float(data[2]) 
    bbox_height = float(data[3])
    center_x = float(data[0])
    center_y = float(data[1])
    out[0] = (center_x - (bbox_width / 2))
    out[1] = (center_y - (bbox_height / 2))
    out[2] = (center_x + (bbox_width / 2))
    out[3] = (center_y + (bbox_height / 2))
    return out


def xyxy2center_width_height(data):
    out = np.zeros((4,))
    bbox_width = data[2] - data[0]
    bbox_height = data[3] - data[1]
    center_x = data[0] + (bbox_width / 2)
    center_y = data[1] + (bbox_height / 2)
    out[0] = center_x
    out[1] = center_y
    out[2] = bbox_width
    out[3] = bbox_height
    return np.array(out)


class Bbox:

    def __init__(self, x1, y1, x2, y2, class_id, score, parent=None, class_name=None):
        self.x1 = x1 # left top
        self.y1 = y1 # left top
        self.x2 = x2 # right bottom
        self.y2 = y2 # right bottom
        self.class_id = int(class_id)
        self.score = score
        self.parent = parent
        if class_name is None:
            self.class_name = classes[int(class_id)]
        else:
            self.class_name = class_name
        
    def __repr__(self):
        return 'Bbox({:.2f}, {:.2f}, {:.2f}, {:.2f}, class_id={}, score={:.2f}, class_name=\'{}\')'.format(self.x1, self.y1, self.x2, self.y2, self.class_id, self.score, self.class_name)

    def xyxy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def yxyx(self):
        return np.array([self.y1, self.x1, self.y2, self.x2])

    def center_width_height(self):
        return xyxy2center_width_height(self.xyxy())

    def cropped_image(self, border=0.0):
        """Return the original image cropped on the bounding box limits
        border: percentage of the bounding box width and height to enlager the bbox
        """
        img = self.parent.img
        h, w = img.shape[:2]
        
        # percentage of bbox dimensions
        bbh, bbw = self.y2 - self.y1, self.x2 - self.x1
        i1 = int(np.max([0, self.y1 - bbh * border / 2.0]))
        i2 = int(np.min([h, self.y2 + bbh * border / 2.0]))
        j1 = int(np.max([0, self.x1 - bbw * border / 2.0]))
        j2 = int(np.min([w, self.x2 + bbw * border / 2.0]))
        cropped = img[i1: i2, j1: j2]

        # # percentage of total image dimensions
        # i1 = int(np.max([0, self.y1 - h * border / 2.0]))
        # i2 = int(np.min([h, self.y2 + h * border / 2.0]))
        # j1 = int(np.max([0, self.x1 - w * border / 2.0]))
        # j2 = int(np.min([w, self.x2 + w * border / 2.0]))
        # cropped = img[i1: i2, j1: j2]

        return cropped

    
    def draw(self, img=None):
        if img is None:
            img = self.parent.img
        color = CMAP(self.class_id/(len(classes) - 1))[:3]
        color = [255 * c for c in color]
        thickness = 1 + int(img.shape[1]/300)
        rimg = cv2.rectangle(
            img, (self.x1, self.y1), (self.x2, self.y2), color, thickness=thickness)
        cv2.putText(
            rimg, self.class_name, (self.x1, self.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return rimg


class BboxList(list):

    def __init__(self, ids=None, scores=None, bboxes=None, th=0.0, img=None):
        super(BboxList, self).__init__()
        self.img = img
        self.th = th
        if ids is not None:
            self.initialize(ids, scores, bboxes)

    def initialize(self, ids, scores, bboxes):
        out_bboxes = []
        for _id, score, bbox in zip(ids, scores, bboxes):
            if score > self.th: 
                out_bboxes.append(Bbox(bbox[0], bbox[1], bbox[2], bbox[3], _id, score, parent=self))
        out_bboxes = sorted(out_bboxes, key=(lambda x: x.score), reverse=True)
        self.extend(out_bboxes)

    def to_arrays(self):
        ids = []
        scores = []
        bboxes = []
        for bbox in self:
            ids.append(bbox.class_id)
            scores.append(bbox.score)
            bboxes.append(bbox.xyxy())
        return np.array(ids), np.array(scores), np.array(bboxes)


    def draw(self, img=None):
        if img is None:
            img = self.img

        for bbox in self:
            img = bbox.draw(img)

        return img


if __name__ == '__main__':

    pass