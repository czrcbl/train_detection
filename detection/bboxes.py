from __future__ import division, print_function, absolute_import
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


class Bbox(object):

    def __init__(self, x1, y1, x2, y2, class_id, score, class_name, parent=None):
        self.x1 = x1 # left top
        self.y1 = y1 # left top
        self.x2 = x2 # right bottom
        self.y2 = y2 # right bottom
        self.class_id = int(class_id)
        self.score = score
        self.parent = parent
        self.class_name = class_name
        
    def __repr__(self):
        return 'Bbox({:.2f}, {:.2f}, {:.2f}, {:.2f}, class_id={}, score={:.2f}, class_name=\'{}\')'.format(self.x1, self.y1, self.x2, self.y2, self.class_id, self.score, self.class_name)

    @property
    def xyxy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @property
    def yxyx(self):
        return np.array([self.y1, self.x1, self.y2, self.x2])

    @property
    def center_width_height(self):
        return xyxy2center_width_height(self.xyxy)

    def cropped_image(self, img, border=0.0):
        """Return the original image cropped on the bounding box limits
        border: percentage of the bounding box width and height to enlager the bbox
        """
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

    def mask_image(self, img=None):

        if img is None:
            img = self.parent.img
        
        if len(img.shape) == 2:
            h, w = img.shape
            out = np.zeros((h, w))
        elif len(img.shape) == 3:
            h, w, c = img.shape
            out = np.zeros((h, w, c))

        bbh, bbw = self.y2 - self.y1, self.x2 - self.x1
        i1 = int(np.max([0, self.y1]))
        i2 = int(np.min([h, self.y2]))
        j1 = int(np.max([0, self.x1]))
        j2 = int(np.min([w, self.x2]))
        out[i1: i2, j1: j2] = img[i1: i2, j1: j2]
        # out = img[i1: i2, j1: j2, :]
        return out
    
    
    def draw(self, img):
        """Draw bbox on image, expect an int image"""
        img = np.copy(img)
        height, width = img.shape[:2]
        color = plt.get_cmap('hsv')(self.class_id / len(self.parent.classes))
        color = [x * 255 for x in color]
        thickness = 1 + int(img.shape[1]/300)
        cv2.rectangle(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, thickness)
        text = '{} {:d}%'.format(self.class_name, int(self.score * 100))
        font_scale = 0.5/600 * width
        thickness = int(2/600 * width)
        vert = 10/1080 * height
        cv2.putText(img, text, (int(self.x1), int(self.y1 - vert)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return img


class BboxList(list):

    def __init__(self, ids, scores, bboxes, class_names, th=0.0):
        super(BboxList, self).__init__()
        self.th = th
        self.classes = class_names
        self.initialize(ids, scores, bboxes)

    def initialize(self, ids, scores, bboxes):
        out_bboxes = []
        for _id, score, bbox in zip(ids, scores, bboxes):
            _id = int(_id)
            if score > self.th: 
                out_bboxes.append(Bbox(bbox[0], bbox[1], bbox[2], bbox[3], _id, score, self.classes[_id], parent=self))
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

    def draw(self, img):

        # inverse order to focos on higher score boxes
        for bbox in self[::-1]:
            img = bbox.draw(img)

        return img


if __name__ == '__main__':

    pass