from os.path import join as pjoin

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric

from .utils import RealDataset, SynthDataset, RealGraspDataset, SpecRealDataset, SpecSynthDataset
from .transforms import SSDValTransform, SSDTrainTransform
from . import config as cfg


def get_dataset(dataset, args):

    if dataset.lower() == 'real':
        train_dataset = RealDataset(mode='train')
        val_dataset = RealDataset(mode='test')
    elif dataset.lower() == 'real_with_grasp':
        train_dataset = RealGraspDataset(mode='train')
        val_dataset = RealGraspDataset(mode='test')
    elif dataset.lower() == 'synth_spec':
        train_dataset = SpecSynthDataset(tclass=args.tclass, root=pjoin(cfg.dataset_folder, 'synth_small_bg'), mode='all')
        val_dataset = SpecRealDataset(tclass=args.tclass, mode='all')
    elif dataset.lower() == 'synth_small_printer':
        train_dataset = SynthDataset(root=pjoin(cfg.dataset_folder, 'synth_small_printer'), mode='all')
        val_dataset = RealDataset(mode='test')
    elif dataset.lower() == 'synth_part2':
        train_dataset = SpecSynthDataset(root=pjoin(cfg.dataset_folder, 'synth_small_printer'), tclass='part2', mode='all')
        val_dataset = SpecRealDataset(mode='all', tclass='part2')
    elif dataset.lower() == 'synth_part3':
        train_dataset = SpecSynthDataset(root=pjoin(cfg.dataset_folder, 'synth_small_printer'), tclass='part3', mode='all')
        val_dataset = SpecRealDataset(mode='all', tclass='part3')
    elif dataset.lower() == 'synth_dosing_nozzle':
        train_dataset = SpecSynthDataset(root=pjoin(cfg.dataset_folder, 'synth_small_printer'), tclass='part3', mode='all')
        val_dataset = SpecRealDataset(mode='all', tclass='dosing_nozzle')
    elif dataset.split('_')[0] == 'synth':
        train_dataset = SynthDataset(root=pjoin(cfg.dataset_folder, dataset), mode='all')
        val_dataset = RealDataset(mode='test')


    val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    
    if args.mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)

    return train_dataset, val_dataset, val_metric


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    # if save_interval and (epoch + 1) % save_interval == 0:
    #     logger.info('[Epoch {}] Saving parameters to {}'.format(
    #         epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
    #     net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def validate_ssd(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()
    
def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))