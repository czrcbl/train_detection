from __future__ import division, print_function, absolute_import

import os
from os.path import join as pjoin
import numpy as np

import mxnet as mx
from mxnet import nd
from gluoncv import model_zoo
from gluoncv.data.transforms import presets

from . import config as cfg
from . import transforms
from . bboxes import BboxList


def filter_predictions(ids, scores, bboxes, threshold=0.0):
    """filter and resize predictions"""
    idx = scores.squeeze().asnumpy() > threshold
    fscores = scores.squeeze().asnumpy()[idx]
    fids = ids.squeeze().asnumpy()[idx]
    fbboxes = bboxes.squeeze().asnumpy()[idx]
    return fids, fscores, fbboxes

# (width, height) 
# trans_map = {
#     'ssd512': transforms.SSDDefaultTransform(512, 512),
#     'ssd300': transforms.SSDDefaultTransform(300, 300),
#     'yolo:': 0
# }


def list_models():
    
    data = {}
    base_dir = cfg.checkpoints_folder
    datasets = [x for x in sorted(os.listdir(base_dir)) if os.path.isdir(pjoin(base_dir, x))]
    for ds in datasets:
        models = os.listdir(pjoin(base_dir, ds))
        data[ds] = models

    return data

def load_model(model, dataset):
    pass


class Detector:
    model_data = list_models()
    def __init__(self, model='ssd512', dataset='real', ctx='cpu', classes=cfg.classes, model_path=None):
        data = Detector.model_data
        if dataset not in data.keys():
            raise ValueError('Dataset {} does not exist, avaliable datasets:{}'.format(dataset, data.keys()))
        elif model not in data[dataset]:
            raise ValueError('Model {} does not exist for dataset {}, avaliable models:{}'.format(model, dataset, data.keys()))
        
        #dataset_root = pjoin(cfg.dataset_folder, dataset)
        # with open(pjoin(dataset_root, 'classes.txt'), 'r') as f:
        #     classes = [line.strip() for line in f.readlines()]
        #     classes = [line for line in classes if line]
        
        self.classes = classes
        self.model = model
        if ctx == 'cpu':
            ctx = mx.cpu()
        elif ctx == 'gpu':
            ctx = mx.gpu()
        else:
            raise ValueError('Invalid context.')
        self.ctx = ctx
        self.short, self.width, self.height = None, None, None
        if model.lower() == 'ssd512':
            model_name = 'ssd_512_resnet50_v1_coco'
            parameters_path = pjoin(cfg.checkpoints_folder, dataset, 'ssd512/transfer_512_ssd_512_resnet50_v1_coco_best.params')
            self.width, self.height = 512, 512
            self.transform = transforms.SSDDefaultTransform(self.width, self.height)
        elif model.lower() == 'ssd300':
            model_name = 'ssd_300_vgg16_atrous_coco'
            parameters_path = pjoin(cfg.checkpoints_folder, dataset, 'ssd300/transfer_300_ssd_300_vgg16_atrous_coco_best.params')
            self.width, self.height = 300, 300
            self.transform = transforms.SSDDefaultTransform(self.width, self.height)
        elif (model.lower() == 'yolo') or (model.lower() == 'yolo416'):
            model_name = 'yolo3_darknet53_coco'
            parameters_path = pjoin(cfg.checkpoints_folder, dataset, 'yolo416/transfer_416_yolo3_darknet53_coco_best.params')
            self.width, self.height = 416, 416
            self.transform = transforms.SSDDefaultTransform(self.width, self.height)
        elif (model.lower() == 'frcnn') or (model.lower() == 'faster_rcnn'):
            model_name = 'faster_rcnn_resnet50_v1b_coco'
            parameters_path = pjoin(cfg.checkpoints_folder, dataset, 'faster_rcnn/transfer_faster_rcnn_resnet50_v1b_coco_best.params')
            self.short = 600
            self.transform = transforms.FasterRCNNDefaultTransform(short=600)
        elif model.lower() == 'ssd512_mobile':
            model_name = 'ssd_512_mobilenet1.0_coco'
            parameters_path = pjoin(cfg.checkpoints_folder, dataset, 'ssd512_mobile/transfer_512_ssd_512_mobilenet1.0_coco_best.params')
            self.width, self.height = 512, 512
            self.transform = transforms.SSDDefaultTransform(self.width, self.height)
        else:
            raise ValueError('Invalid model `{}`.'.format(model.lower()))

        net = model_zoo.get_model(model_name, pretrained=False, ctx=ctx)
        net.initialize(force_reinit=True, ctx=ctx)
        net.reset_class(classes=classes)
        if model_path is not None:
            parameters_path = model_path
        net.load_parameters(parameters_path, ctx=ctx)
        self.net = net


    @classmethod
    def list_datasets(cls):
        return cls.model_data.keys()


    @classmethod
    def list_models(cls, dataset):
        try:
            models = cls.model_data[dataset]
        except KeyError:
            raise ValueError('Dataset {} does not exist, avaliable datasets {}'.format(dataset, cls.model_data.keys()))
        return models


    def detect(self, img, threshold=0.5, mantain_scale=True):
        """ 
        Detects Bounding Boxes in a image.
        Inputs
        ------
        img: input image as a numpy array
        threshold: detection threshold
        mantain_sacale: if true return bounding boxes in the original image coordinates
        
        Outputs
        -------
        bbox_list: a bounding box list object containing all filtered predictions
        timg: transformed image
        """
        # TODO: improve this check to work in all cases
        if np.max(img) < 1.1:
            img = img * 255
        
        in_height, in_width = img.shape[:2]

        timg = self.transform(mx.nd.array(img))
        t_height, t_width = timg.shape[1:]

        width_ratio = in_width / t_width
        height_ratio = in_height / t_height

        timg = self.transform(mx.nd.array(img))
        ids, scores, bboxes = self.net(timg.expand_dims(axis=0).as_in_context(self.ctx))
        fids, fscores, fbboxes = filter_predictions(ids, scores, bboxes, 
            threshold=threshold)
        if mantain_scale:
            rep = np.repeat(
                np.array([[width_ratio, height_ratio, width_ratio, height_ratio]]),
                fbboxes.shape[0], axis=0)
            rscaled_bboxes = fbboxes * rep
            out_img = img
        else:
            rscaled_bboxes = fbboxes
            out_img = timg
        box_list = BboxList(fids, fscores, rscaled_bboxes, self.classes, th=threshold, img=out_img)
        return box_list, timg