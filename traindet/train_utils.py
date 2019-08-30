from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

from traindet.utils import RealDataset, SynthDataset


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
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    
    if mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)

    return train_dataset, val_dataset, val_metric