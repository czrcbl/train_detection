from gluoncv.utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric

from traindet.utils import RealDataset, SynthDataset, RealGraspDataset


def get_dataset(dataset, mixup=False):

    if dataset.lower() == 'real':
        train_dataset = RealDataset(mode='train')
        val_dataset = RealDataset(mode='test')   
    elif dataset.lower() == 'synth':
        train_dataset = SynthDataset(mode='train')
        val_dataset = SynthDataset(mode='test')
    elif dataset.lower() == 'mixed':
        train_dataset = SynthDataset(mode='all')
        val_dataset = RealDataset(mode='all')
    elif dataset.lower() == 'synth02':
        train_dataset = SynthDataset(root='datasets/synth02', mode='all')
        val_dataset = RealDataset(mode='all')
    elif dataset.lower() == 'real_with_grasp':
        train_dataset = RealGraspDataset(mode='train')
        val_dataset = RealGraspDataset(mode='test')
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    val_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    
    if mixup:
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
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))