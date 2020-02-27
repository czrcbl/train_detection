import _fix_path
import os
import argparse
from os.path import join as pjoin
from traindet.utils import load_map
from traindet import config as cfg
from traindet.train_utils import get_dataset

import mxnet as mx
from mxnet import gluon
from gluoncv.data import transforms
from gluoncv import model_zoo
from gluoncv.model_zoo import get_model
# def parse_args():

#     parser = argparse.ArgumentParser('Get a report on the performance of the \
#         input model on the input dataset.')
#     parser.add_argument('--model')
#     parser.add_argument('--dataset')

#     args = parser.parse_args()

#     return args


# def main():

#     args = parse_args()
#     map_score, epoch = load_map(args.model, args.dataset)

#     print(map_score)
#     print(f'\nMax map reached on epoch:\n')
#     print(epoch)

def parse_args():

    parser = argparse.ArgumentParser('Validate model on a dataset')
    parser.add_argument('--type', help='SSD, YOLO or FRCNN')
    parser.add_argument('--model', help='Model name.')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--store-predictions', action='store_true',
        help='whether to save the predictions.')

    args = parser.parse_args()

    return args

def parse_model(model_path):

    mfiles = os.listdir(model_path)
    for fname in mfiles:
        if os.path.splitext(fname)[-1] == '.params':
            params_file = fname
    
    log_file = params_file[:-11] + 'train.log'

    with open(pjoin(model_path, log_file), 'r') as f:
        line = f.readlines()[0]
    line = line.replace('Namespace', 'dict')
    print(line)
    params_dict = eval(line)
    return params_file, params_dict


# def load_model(model_type, params_path, base_model, ctx=mx.gpu()):

#     if model_type == 'ssd300':
#         net = model_zoo.get_model(base_model, pretrained=True, ctx=ctx, prefix='ssd0_')
#         transform = transforms.presets.ssd.SSDDefaultValTransform(width=300, height=300)
#     elif model_name == 'ssd512':
#         net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=ctx, prefix='ssd0_')
#         params_path = pjoin(cfg.project_folder, f'data/checkpoints/{dataset}/ssd512/transfer_512_ssd_512_resnet50_v1_coco_best.params')
#         transform = transforms.presets.ssd.SSDDefaultValTransform(width=512, height=512)
#     elif model_name == 'yolo416':
#         net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)
#         params_path = pjoin(cfg.project_folder, f'data/checkpoints/{dataset}/yolo416/transfer_416_yolo3_darknet53_coco_best.params')
#         transform = transforms.presets.yolo.YOLO3DefaultValTransform(width=416, height=416)
#     elif model_name == 'frcnn':
#         net = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True, ctx=ctx)
#         params_path = pjoin(cfg.project_folder, f'data/checkpoints/{dataset}/faster_rcnn/transfer_faster_rcnn_resnet50_v1b_coco_best.params')
#         transform = transforms.presets.rcnn.FasterRCNNDefaultValTransform(short=600)
#     else:
#         raise NotImplementedError(f'Model {model_name} is not implemented.')
        
#     net.reset_class(classes=cfg.classes)
#     net.initialize(force_reinit=True, ctx=ctx)
#     net.load_parameters(params_path, ctx=ctx)

#     return net, transform

def main():

    args = parse_args()
    model_path = pjoin(cfg.checkpoints_folder, args.dataset, args.model)

    params_file, params_dict = parse_model(model_path)
    
    ctx = mx.gpu()    
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset)

    net = get_model(params_dict['base_model'], pretrained=True, norm_layer=gluon.nn.BatchNorm)
    net.reset_class(classes=train_dataset.classes)
    net.initialize(force_reinit=True)
    net.load_parameters(pjoin(model_path, params_file))

    async_net = net

    if args.type.lower() == 'ssd':
        train_data, val_data = get_ssd_dataloader(
            async_net, train_dataset, val_dataset, params_dict['data_shape'], params_dict['batch_size'], 
            args.num_workers, bilateral_kernel_size=args.bilateral_kernel_size,
            sigma_vals=args.sigma_vals, grayscale=args.grayscale)

if __name__ == '__main__':
    main()

