import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
if path not in sys.path:
    sys.path.insert(0, path)
import unittest

from detection import Detector
from detection import config as dcfg
from ggcnn_detection import GDetector



class Tests(unittest.TestCase):

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


if __name__ == '__main__':

    unittest.main()