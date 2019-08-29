import mxnet as mx
import mxnet.gluon.data as gdata
from mxnet import nd
# from gluoncv.utils import viz
from gluoncv import model_zoo
from gluoncv.data import transforms
# from gluoncv.utils import bbox_iou
# from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
import random
import os
from os.path import join as pjoin
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


class RealDataset(gdata.Dataset):
    
    def __init__(self, root=cfg.real_dataset_folder, train=True):
        super(RealDataset, self).__init__()
        classes = []
        with open(pjoin(root, 'classes.txt'), 'r') as f:
            for line in f:
                classes.append(line.strip())
        self.classes = classes
        if train:
            images_dir = pjoin(root, 'train')
        else:
            images_dir = pjoin(root, 'test')
        files = sorted(os.listdir(images_dir))
        img_fns  = [fn for fn in files if fn.split('.')[-1] == 'png']
        names = [s.split('.')[0] for s in img_fns]
        target = []
        for name in names:
            path = pjoin(images_dir, name + '.txt')
            with open(path, 'r') as f:
                labels = []
                for line in f:
                    data = [float(s) for s in line.strip().split(' ')]
                    label = yolo2voc(data)
                    labels.append(label)
            target.append(np.array(labels))
        
        assert(len(img_fns) == len(target))
        img_fns = [pjoin(images_dir, fn) for fn in img_fns]
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


def load_model(model_name, ctx=mx.gpu()):

    if model_name == 'ssd300':
        net = model_zoo.get_model('ssd_300_vgg16_atrous_coco', pretrained=True, ctx=ctx, prefix='ssd0_')
        # net = model_zoo.get_model('ssd_300_vgg16_atrous_coco', pretrained=True, ctx=ctx)
        params_path = pjoin(cfg.project_folder, 'checkpoints/ssd300/transfer_300_ssd_300_vgg16_atrous_coco_best.params')
        transform = transforms.presets.ssd.SSDDefaultValTransform(width=300, height=300)
    elif model_name == 'ssd512':
        net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=ctx, prefix='ssd0_')
        params_path = pjoin(cfg.project_folder, 'checkpoints/ssd512/transfer_512_ssd_512_resnet50_v1_coco_best.params')
        transform = transforms.presets.ssd.SSDDefaultValTransform(width=512, height=512)
    elif model_name == 'yolo416':
        net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)
        params_path = pjoin(cfg.project_folder, 'checkpoints/yolo416/transfer_416_yolo3_darknet53_coco_best.params')
        transform = transforms.presets.yolo.YOLO3DefaultValTransform(width=416, height=416)
    elif model_name == 'frcnn':
        net = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True, ctx=ctx)
        params_path = pjoin(cfg.project_folder, 'checkpoints/faster_rcnn/transfer_faster_rcnn_resnet50_v1b_coco_best.params')
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


if __name__ == '__main__':

    prefix = 'checkpoints/ssd512/transfer_512_ssd_512_resnet50_v1_coco'

    epo, out = parse_log(prefix)

    print(epo, out)