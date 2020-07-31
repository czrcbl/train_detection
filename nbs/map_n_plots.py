# %% [markdown]
# # mAP Table and Visualizations
# This notebook generates the comparative mAP table and the visualizatios for the paper.
#
# %%
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# %%
import sys
sys.path.insert(0, '../')

# %%
import os
from os.path import join as pjoin
from collections import OrderedDict
import cv2
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from gluoncv.utils import viz

from traindet.train_utils import get_dataset
from traindet.utils import load_predictions
import traindet.config as cfg
from traindet.utils import calc_map, parse_log, map_history

from gluoncv.utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

from copy import deepcopy as copy
from easydict import EasyDict as edict
from sklearn import metrics
from sklearn.metrics import classification_report
from traindet import get_class_preds, build_confusion_matrix, calc_detection
import seaborn as sns
sns.set_style('whitegrid')

# %%
article_root = pjoin(cfg.data_folder, 'article')
out_folder = pjoin(article_root, 'tables')
outimg_folder = pjoin(article_root, 'figures')

#%%
_, val_ds_real, _ = get_dataset('real')
_, val_ds, _ = get_dataset('synth_small_bg')
print('Validation set sizes:', len(val_ds), len(val_ds_real))

# %%
models = 'ssd_default,yolo_default,frcnn_default'.split(',')
datasets = 'real,synth_small_printer,synth_small_nobg'.split(',')

datasets_form = ['Real','Synth', 'Synth No Bg.']
models_form = [ 'FRCNN', 'SSD', 'YOLOv3']

# %%
data = load_predictions('ssd_default,yolo_default,frcnn_default','real,synth_small_printer,synth_small_nobg')

# %%
def save_text(text, filepath):
    with open(filepath, 'w') as f:
        f.write(text)
        
# %%
size_map = {
    'frcnn_default': (800, 600),
    'ssd_default': (512,512),
    'yolo_default': (608, 608)
}
def calc_map_all(data, size_map, metric):
    out = dict()
    for dataset in data.keys():
        out[dataset] = dict()
        for model in data[dataset].keys():
            out[dataset][model] = dict()
            mAP = calc_map(data[dataset][model].preds, data[dataset][model].labels, (800, 600), metric)
            for cls_, val in zip(*mAP):
                out[dataset][model][cls_] = val
    return out

# %%
metric = VOCMApMetric(iou_thresh=0.5, class_names=val_ds.classes)
mAP_map = calc_map_all(data, size_map, metric)
# %%
out = mAP_map
df = pd.DataFrame({(i,j): out[i][j] for i in datasets for j in models})
df

# %%
datasets_form0 = [datasets_form[0], datasets_form[2], datasets_form[1]]
df.index = cfg.abre_classes + ['mAP']
df.columns.set_levels(datasets_form0,level=0, inplace=True)
df.columns.set_levels(models_form, level=1, inplace=True)
df

# %%
# save_text((df * 100).round(1).to_latex(), out_folder + '/map_table_test.txt')
# %%
log_map = {
'ssd_default': 'transfer_512_ssd_512_resnet50_v1_coco_train.log',
'frcnn_default': 'transfer_faster_rcnn_resnet50_v1b_coco_train.log',
'yolo_default': 'transfer_608_yolo3_darknet53_coco_train.log'
}
hists = edict()
for dataset in datasets:
    hists[dataset] = edict()
    for model in models:
        hists[dataset][model] = map_history(pjoin(cfg.checkpoints_folder, dataset, model, log_map[model]))

epdata = pd.DataFrame(data=hists['real'], columns=['ssd_default', 'yolo_default', 'frcnn_default'])
epdata.columns = ['SSD', 'YOLOv3', 'FRCNN']
fig, ax = plt.subplots(figsize=(4,3))
sns.lineplot(data=epdata, palette="tab10", linewidth=2.5, ax=ax)
ax.set_xlabel('Epoch')
ax.set_ylabel('mAP')
ax.set_ylim([0, 1.1])
# %%
preds = data.synth_small_printer.frcnn_default['preds']
labels = data.synth_small_printer.frcnn_default['labels']
print(len(preds), len(labels))

# %%
def plot_example(img, datasets, label, preds, axs, threshold, classes):
        
    for i, (dataset, ax) in enumerate(zip(datasets, axs)):
        rimg = img
#     sizes = (800, 600)
#     rimg = cv2.resize(img, dsize=sizes[model], interpolation=cv2.INTER_NEAREST)
        pred = preds[dataset]
        ax = viz.plot_bbox(rimg, pred[:, :4], scores=pred[:, 5], labels=pred[:, 4], class_names=classes, ax=ax, thresh=threshold)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="b", lw=2)
            ax.text(20, 60, label, fontdict={
                'weight' : 'bold',
                'size'   : 20,
                'color': 'black'},
                bbox=bbox_props
                )

            
            dataset = val_ds_real
# nrows=len(dataset)

real_ids = [0, 6, 10, 33, 93]
synth_ids = [j for i in real_ids for j in range(len(val_ds)) if val_ds_real.fns[i] == val_ds.fns[j]] 
letters = list('abcde')
datasets = ['real', 'synth_small_printer', 'synth_small_nobg']
threshold = 0.5

_, dataset, _ = get_dataset('real')

fig, axs = plt.subplots(
#     nrows=len(real_ids),
    nrows=4,
    ncols=len(datasets), 
    figsize=(18, 21), 
    frameon=False, 
    gridspec_kw = {'wspace':0.0125, 'hspace':0.025})

for i, (n, m, ax) in enumerate(zip(real_ids, synth_ids, axs)):
    if i == 0:
        for ax1, name in zip(ax, datasets_form):
            ax1.set_title(name)
    preds = {ds: data[ds]['frcnn_default']['preds'][m] for ds in ['synth_small_printer', 'synth_small_nobg']}
    preds['real'] = data['real']['frcnn_default']['preds'][n]
    img = dataset[n][0].asnumpy()
    plot_example(img, datasets, letters[i], preds, ax, threshold, cfg.classes)
    
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.show()
# %%

