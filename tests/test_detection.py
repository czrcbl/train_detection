import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
if path not in sys.path:
    sys.path.insert(0, path)
import unittest

import numpy as np

from detection import Detector
from detection.bboxes import BboxList, BboxList
from detection import config as dcfg
from ggcnn_detection import GDetector



class TestDetector(unittest.TestCase):

    def test_load(self):

        ava_datasets = Detector.list_datasets()
        for dataset_name in ava_datasets:
            models = Detector.list_models(dataset_name)
            for model_name in models:
                try:
                    classes = dcfg.classes
                    det = Detector(model=model_name, dataset=dataset_name, ctx='cpu', classes=classes)
                except:
                    classes = dcfg.classes_grasp
                    det = Detector(model=model_name, dataset=dataset_name, ctx='cpu', classes=classes)
        
        gdet = GDetector()


class TestBBoxes(unittest.TestCase):

    def test_loader(self):
        ids = np.array([1, 2, 3])
        scores = np.array([0.5, 0.7, 0.6])
        bboxes = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])
        class_names = list('abcd')
        bblist = BboxList(ids, scores, bboxes, class_names)

        img = np.zeros((100, 100, 3))

        dimg = bblist.draw(img)


if __name__ == '__main__':

    unittest.main()