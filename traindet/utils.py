import mxnet as mx
import mxnet.gluon.data as gdata
from mxnet import nd
# from gluoncv.utils import viz
import gluoncv as gcv
from gluoncv import model_zoo
from gluoncv.data import transforms
# from gluoncv.utils import bbox_iou
# from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
import pandas as pd
import random
import json
import os
import pickle
from os.path import join as pjoin
from copy import copy
from easydict import EasyDict as edict
from traindet import config as cfg


def yolo2voc(data):
    """Convert yolo format to voc, both in relative coordinates."""
    voc = []
    bbox_width = float(data[3]) 
    bbox_height = float(data[4])
    center_x = float(data[1])
    center_y = float(data[2])
    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))
    voc.append(data[0])
    return voc


def load_targets(files_list):
    target = []
    for path in files_list:
        with open(path, 'r') as f:
            labels = []
            for line in f:
                data = [float(s) for s in line.strip().split(' ')]
                label = yolo2voc(data)
                labels.append(label)
        target.append(np.array(labels))
    
    return target


def process_examples(fn, tgt):

        img = np.array(Image.open(fn))
        heigt, width = img.shape[:2]
        ntgt = np.zeros(shape=tgt.shape)
        ntgt[:, [0, 2]] = tgt[:, [0, 2]] * width
        ntgt[:, [1, 3]] = tgt[:, [1, 3]] * heigt
        ntgt[:, 4] = tgt[:, 4]

        return nd.array(img), np.array(ntgt)


class RealDataset(gdata.Dataset):
    
    def __init__(self, root=cfg.real_dataset_folder, mode='train'):
        super(RealDataset, self).__init__()
        self.classes = cfg.classes
        # with open(pjoin(root, 'classes.txt'), 'r') as f:
        #     for line in f:
        #         classes.append(line.strip())
        if mode == 'train':
            images_dir = [pjoin(root, 'train')]
        elif mode == 'test':
            images_dir = [pjoin(root, 'test')]
        elif mode == 'all':
            images_dir = [pjoin(root, 'train'), pjoin(root, 'test')]

        files = []
        for img_dir in images_dir:
            files.extend([pjoin(img_dir, fn) for fn in sorted(os.listdir(img_dir))])

        img_fns  = [pjoin(root, fn) for fn in files if fn.split('.')[-1] == 'png']
        names = [s.split('.')[0] for s in img_fns]
        target = []
        for name in names:
            path = name + '.txt'
            with open(path, 'r') as f:
                labels = []
                for line in f:
                    data = [float(s) for s in line.strip().split(' ')]
                    label = yolo2voc(data)
                    labels.append(label)
            target.append(np.array(labels))
        
        assert(len(img_fns) == len(target))
        self.fns = img_fns
        self.target = target
                  
    def __len__(self):
        return len(self.fns)
                  
    def __getitem__(self, idx):
        """The class id is the -1 element.
            BBoxes coordinaes are the 4 first elements, in the order:
            top left x, top left y, bottom right x, bottom right y
            x is the horizontal coordinate (column)
            y is the vertical coordinate (row)
        """
        fn = self.fns[idx]
        tgt = self.target[idx]
        img = np.array(Image.open(fn))
        heigt, width = img.shape[:2]
        ntgt = np.zeros(shape=tgt.shape)
        ntgt[:, [0, 2]] = tgt[:, [0, 2]] * width
        ntgt[:, [1, 3]] = tgt[:, [1, 3]] * heigt
        ntgt[:, 4] = tgt[:, 4]
        return nd.array(img), np.array(ntgt)


class RealGraspDataset(gdata.Dataset):
    def __init__(self, root=cfg.real_grasp_dataset_folder, mode='train'):
        super(RealGraspDataset, self).__init__()
        self.classes = cfg.classes_grasp
        if mode == 'train':
            files = ['train.json']
        elif mode == 'test':
            files = ['test.json']
        elif mode == 'all':
            files = ['train.json', 'test.json']
        else:
            raise ValueError(f'Invalid mode {mode}.')
        data = []
        for fn in files:
            with open(pjoin(root, fn), 'r') as f:
                data.extend(json.load(f))

        self.fns = [pjoin(root, d['img']) for d in data]
        self.targets = load_targets([pjoin(root, d['label']) for d in data])
    
    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        return process_examples(self.fns[idx], self.targets[idx])


class SynthDataset(gdata.Dataset):
    def __init__(self, root=cfg.synth_dataset_folder, mode='train'):
        super(SynthDataset, self).__init__()
        self.classes = cfg.classes
        if mode == 'train':
            files = ['train.json']
        elif mode == 'test':
            files = ['test.json']
        elif mode == 'all':
            files = ['train.json', 'test.json']
        else:
            raise ValueError(f'Invalid mode {mode}.')
        data = []
        for fn in files:
            with open(pjoin(root, fn), 'r') as f:
                data.extend(json.load(f))

        self.fns = [pjoin(root, d['img']) for d in data]
        self.targets = load_targets([pjoin(root, d['label']) for d in data])
    
    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        return process_examples(self.fns[idx], self.targets[idx])


def load_model(model_type, model_name, dataset, ctx=mx.gpu()):

    if model_type == 'ssd300':
        net = model_zoo.get_model('ssd_300_vgg16_atrous_coco', pretrained=True, ctx=ctx, prefix='ssd0_')
        # net = model_zoo.get_model('ssd_300_vgg16_atrous_coco', pretrained=True, ctx=ctx)
        params_path = pjoin(cfg.project_folder, f'data/checkpoints/{dataset}/{model_name}/transfer_300_ssd_300_vgg16_atrous_coco_best.params')
        transform = transforms.presets.ssd.SSDDefaultValTransform(width=300, height=300)
    elif model_type == 'ssd512':
        net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=ctx, prefix='ssd0_')
        params_path = pjoin(cfg.project_folder, f'data/checkpoints/{dataset}/{model_name}/transfer_512_ssd_512_resnet50_v1_coco_best.params')
        transform = transforms.presets.ssd.SSDDefaultValTransform(width=512, height=512)
    elif model_type == 'yolo416':
        net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)
        params_path = pjoin(cfg.project_folder, f'data/checkpoints/{dataset}/{model_name}/transfer_416_yolo3_darknet53_coco_best.params')
        transform = transforms.presets.yolo.YOLO3DefaultValTransform(width=416, height=416)
    elif model_type == 'frcnn':
        net = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True, ctx=ctx)
        params_path = pjoin(cfg.project_folder, f'data/checkpoints/{dataset}/{model_name}/transfer_faster_rcnn_resnet50_v1b_coco_best.params')
        transform = transforms.presets.rcnn.FasterRCNNDefaultValTransform(short=600)
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented.')
        
    net.reset_class(classes=cfg.classes)
    net.initialize(force_reinit=True, ctx=ctx)
    net.load_parameters(params_path, ctx=ctx)

    return net, transform


def parse_log(prefix):

    n = len(cfg.classes)
    epoch_map_file = prefix + '_best_map.log'
    epochs = []
    scores = []
    with open(epoch_map_file, 'r') as f:
        for line in f:
            l = line.split(':')
            epochs.append(int(l[0].strip()))
            scores.append(float(l[1].strip()))

    epo = epochs[np.array(scores).argmax()]
    log_file = prefix + '_train.log'
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == f'[Epoch {epo}] Validation:':
                break
    l = lines[i + 1: i + n + 2]
    out = {}
    for e in [o.strip().split('=') for o in l]:
        out[e[0]] = float(e[1])
    # out = dict(exec(''.join(out)))

    return epo, out


def load_map(model, dataset):
    
    prefixes = {
        'ssd300': f'{dataset}/ssd300/transfer_300_ssd_300_vgg16_atrous_coco',
        'ssd512': f'{dataset}/ssd512/transfer_512_ssd_512_resnet50_v1_coco',
        'yolo416': f'{dataset}/yolo416/transfer_416_yolo3_darknet53_coco',
        'frcnn': f'{dataset}/faster_rcnn/transfer_faster_rcnn_resnet50_v1b_coco'
    }
    if model == 'all':
        models = cfg.model_names
    else:
        models = [model]

    # paths = [pjoin(cfg.project_folder, p) for p in prefixes]
    out = {}
    epochs = {}
    for model in models:
        path = pjoin(cfg.checkpoints_folder, prefixes[model])
        epochs[model], out[model] = parse_log(path)
        
    index = copy(cfg.classes)
    index.append('mAP')
    map_df = pd.DataFrame(out, index=index)
    map_df.columns = models
    index = copy(cfg.formated_classes)
    index.append('mAP')
    map_df.index = index
    
    epoch_df = pd.DataFrame(epochs, index=[0])
    
    return map_df, epoch_df
    

def load_predictions(models, datasets):
    models = models.split(',')
    datasets = datasets.split(',')
    out = edict()
    for dataset in datasets:
        out[dataset] = edict()
        for model in models:
            path = pjoin(cfg.gen_data_folder, f'predictions/{model}_{dataset}.pkl')
            with open(path, 'rb') as f:
                out[dataset][model] = pickle.load(f)

    return out

# def load_predictions(dataset):

#     out = {}
#     for model in cfg.model_names:
#         path = pjoin(cfg.gen_data_folder, f'predictions/{model}_{dataset}.pkl')
#         with open(path, 'rb') as f:
#             out[model] = pickle.load(f)

#     return out


def calc_map(preds, labels, width, height, eval_metric):
    assert len(preds) == len(labels)
    eval_metric.reset()
    # clipper = gcv.nn.bbox.BBoxClipToImage()
    for pred, label in zip(preds, labels):
        label = label[None, :, :]
        pred = pred[None, :, :]
        pred_bboxes = pred[:, :, :4]
        pred_bboxes[:, :, 0] = pred_bboxes[:, :, 0].clip(0, width)
        pred_bboxes[:, :, 2] = pred_bboxes[:, :, 2].clip(0, width)
        pred_bboxes[:, :, 1] = pred_bboxes[:, :, 1].clip(0, height)
        pred_bboxes[:, :, 3] = pred_bboxes[:, :, 3].clip(0, height)
        # pred_bboxes = clipper(nd.array(pred_bboxes), nd.array(img))
        pred_ids = pred[:, :, 4].astype(np.int)
        pred_scores = pred[:, :, 5]
        gt_bboxes = label[:, :, :4]
        gt_ids = label[:, :, 4].astype(np.int)
        eval_metric.update(pred_bboxes, pred_ids, pred_scores, gt_bboxes, gt_ids)
    return eval_metric.get()


if __name__ == '__main__':

    # prefix = 'checkpoints/ssd512/transfer_512_ssd_512_resnet50_v1_coco'
    # epo, out = parse_log(prefix)
    # print(epo, out)
    trn_ds = RealDataset(mode='train')
    test_ds = RealDataset(mode='test')
    all_ds = RealDataset(mode='all')
    print(len(trn_ds), len(test_ds), len(all_ds))
    print(all_ds[0])

    trn_ds = SynthDataset(mode='train')
    test_ds = SynthDataset(mode='test')
    all_ds = SynthDataset(mode='all')

    print(len(trn_ds), len(test_ds), len(all_ds))
    print(all_ds[0])