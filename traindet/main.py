import mxnet as mx
import numpy as np
import pandas as pd
from copy import copy
import subprocess
import pickle
from os.path import join as pjoin
import time
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from gluoncv.utils import bbox_iou

from traindet import config as cfg
from traindet.utils import load_model, RealDataset
from traindet.val_utils import get_val_ssd_dataloader, get_val_frcnn_dataloader

def filter_predictions(ids, scores, bboxes, th=0.5):
    idx = scores.squeeze().asnumpy() > th
    fscores = scores.squeeze().asnumpy()[idx]
    fids = ids.squeeze().asnumpy()[idx]
    fbboxes = bboxes.squeeze().asnumpy()[idx]
    
    return fids, fscores, fbboxes


def get_predictions(net, dataset, transform, ctx=mx.gpu()):
    """Returns a list with a prediction for each image on input dataset and a list
    with the ground truth labels. 
    """
    dataset = dataset.transform(transform)
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    preds = []
    labels = []
    for out in dataset:
        x, label = out[0], out[1]
        ids, scores, bboxes = net(x.expand_dims(axis=0).as_in_context(ctx))
        fscores = scores.squeeze().asnumpy()
        fids = ids.squeeze().asnumpy()
        fbboxes = bboxes.squeeze().asnumpy()
        p = np.concatenate((fbboxes, fids.reshape(-1, 1), fscores.reshape(-1, 1)), axis=1)
        preds.append(p)
        labels.append(label)

    return preds, labels


def predict_and_store(dataset, output_path, ctx=mx.gpu()):
    
    data = {}
    for model_name in cfg.used_model_names:
        net, trans =  load_model(model_name, ctx=ctx)
        preds, labels = get_predictions(net, dataset, trans, ctx)
        data[model_name] = {}
        data[model_name]['predictions'] = preds
        data[model_name]['labels'] = labels
        del net

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


# def get_class_preds(labels, preds, th=0.5, iou_th=0.5):
#     n_classes = len(cfg.classes)
#     ground_truth = []
#     predictions = []
#     for pred, label in zip(preds, labels):
#         for i in range(label.shape[0]):
#             ground_truth.append(label[i, 4])
#             # Make sure that preidiction are ordered from highest to lowest score
#             args = pred[:, 5].argsort()[::-1]
#             opred = pred[args, :] 
#             for j in range(opred.shape[0]):
#                 iou = bbox_iou(label[i, :4].reshape(1, -1), opred[j, :4].reshape(1, -1)).item()
#                 if (opred[j, 5] > th) and (opred[j, 4] == label[i, 4]) and (iou > iou_th):
#                     predictions.append(opred[j, 4])
#                     break
#             else:
#                 predictions.append(n_classes)
    
#     return predictions, ground_truth


def get_class_preds(labels, preds, th=0.5, iou_th=0.5):
    n_classes = len(cfg.classes)
    ground_truth = []
    predictions = []
    for pred, label in zip(preds, labels):
        for i in range(label.shape[0]):
            ground_truth.append(label[i, 4])
            # Make sure that preidiction are ordered from highest to lowest score
            args = pred[:, 5].argsort()[::-1]
            opred = pred[args, :] 
            for j in range(opred.shape[0]):
                iou = bbox_iou(label[i, :4].reshape(1, -1), opred[j, :4].reshape(1, -1)).item()
                if (opred[j, 5] > th) and (iou > iou_th):
                    predictions.append(opred[j, 4])
                    break
            else:
                predictions.append(n_classes)
    
    return predictions, ground_truth


def build_confusion_matrix(ground_truth, predictions, classes):
    classes = copy(classes)
    classes.append('Undetected')
    df = pd.DataFrame(data=confusion_matrix(ground_truth, predictions, labels=np.arange(len(classes))), columns=classes, index=classes)
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df


def calc_detection(labels, preds, th=0.5, iou_th=0.5):
    false_negatives = 0
    false_positives = 0

    for pred, label in zip(preds, labels):

        # args = pred[:, 5].argsort()[::-1]
        # opred = pred[args, :]
        opred = pred[pred[:, 5] > th, :]
        
        for i in range(label.shape[0]):
            for j in range(opred.shape[0]):
                iou = bbox_iou(label[i, :4].reshape(1, -1), opred[j, :4].reshape(1, -1)).item()
                if (opred[j, 4] == label[i, 4]) and (iou > iou_th):
                    break
            else:
                false_negatives += 1

        for i in range(opred.shape[0]): 
            for j in range(label.shape[0]):
                iou = bbox_iou(label[j, :4].reshape(1, -1), opred[i, :4].reshape(1, -1)).item()
                if  (opred[i, 4] == label[j, 4]) and (iou > iou_th):
                    break
            else:
                false_positives += 1

    return false_negatives, false_positives


def benchmark(n_samples, ctx):
    
    dataset = RealDataset(train=False)
    out = {}
    for model in cfg.used_model_names:
        net, transform = load_model(model, ctx=ctx)
        net.hybridize(static_alloc=True, static_shape=True)
        print(f'Testing model {model} on {ctx}.')
        times = []
        for i in range(n_samples):
            img, label = dataset[i]
            tic = time.time()
            t = transform(img, label)
            x, label = t[0], t[1]
            ids, scores, bboxes = net(x.expand_dims(axis=0).as_in_context(ctx))
            mx.nd.waitall()
            times.append(time.time() - tic)
        out[model] = times
        del net
    return out


def benchmark_all(n_samples, output_path):
    out = {} 
    for c, ctx in zip(['GPU', 'CPU'], [mx.gpu(), mx.cpu()]):
        out[c] = benchmark(n_samples, ctx)
    
    with open(output_path, 'wb') as f:
        pickle.dump(out, f)


if __name__ == '__main__':

    val_ds = RealDataset(train=False)
    
    predict_and_store(val_ds, 'gen_data/predictions.pkl')
    
    benchmark_all(20, 'gen_data/benchmarks.pkl')
