"""Train SSD"""
import _fix_path
import argparse
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy

from traindet.train_utils import (get_dataset, save_params, validate_ssd, 
    get_ssd_dataloader)
from traindet.transforms import SSDTrainTransform, SSDValTransform
from gluoncv import model_zoo


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--transfer', action='store_true',
                        help="Whether to apply transfer learning on whole model.")
    parser.add_argument('--base-model', type=str, default='',
                        help="Base model which will be modified.")
    parser.add_argument('--dataset', type=str, default='custom',
                        help='Training dataset. real or synthetic.')
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training mini-batch size')
    parser.add_argument('--dataset_root', type=str, default='',
                        help='Training dataset root, uses default if not provided')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=14, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='60,80',
                        help='epochs at which learning rate decays. default is 20,40.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')

    parser.add_argument('--bilateral-kernel-size', default=None, type=int)
    parser.add_argument('--sigma-vals', default=None, type=int)
    parser.add_argument('--grayscale', action='store_true')

    args = parser.parse_args()
    return args


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    start_train_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        start_epoch_time = time.time()
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate_ssd(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)

        end_epoch_time = time.time()
        logger.info('Epoch time {:.3f}'.format(end_epoch_time - start_epoch_time))
    end_train_time = time.time()
    logger.info('Train time {:.3f}'.format(end_train_time - start_train_time))
    
if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset)

    if args.transfer:
        net_name = f'transfer_{args.data_shape}_{args.base_model}'
    else:
        net_name = f'scratch_{args.data_shape}_{args.base_model}'

    args.save_prefix += net_name

    if args.transfer:
        net = get_model(args.base_model, pretrained=True, norm_layer=gluon.nn.BatchNorm)
    #    net = get_model(args.base_model, pretrained=True)
    else:
        net = get_model(args.base_model, pretrained_base=True, norm_layer=gluon.nn.BatchNorm)
    #   net = get_model(args.base_model, pretrained_base=True)
    
    # net.initialize(force_reinit=True)
    net.reset_class(classes=train_dataset.classes)

    if args.resume.strip():
        net.initialize(force_reinit=True)
        net.load_parameters(args.resume.strip())

    async_net = net
    # training data
    
    train_data, val_data = get_ssd_dataloader(
        async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, 
        args.num_workers, bilateral_kernel_size=args.bilateral_kernel_size,
        sigma_vals=args.sigma_vals, grayscale=args.grayscale)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
    
