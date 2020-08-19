# %%
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# %%
import sys
sys.path.insert(0, '../')
# %%
import numpy as np
import pandas as pd
from copy import copy
from easydict import EasyDict as edict

from sklearn import metrics

from traindet import get_class_preds, build_confusion_matrix
from traindet import config as cfg
from traindet.utils import load_predictions

# %%
data = load_predictions('ssd_default,yolo_default,frcnn_default','real,synth_small_printer,synth_small_nobg')

# %%

target_names = copy(cfg.classes)
target_names.append('Undetected')

# # Calculate class predictions and labels
# class_preds = edict()
# for model, results in data.items():
#     class_preds[model] = edict()
#     class_preds[model].preds, class_preds[model].labels = get_class_preds(results['labels'], results['predictions'], th=0.5, iou_th=0.5)

models = 'ssd_default,yolo_default,frcnn_default'.split(',')
datasets = 'real,synth_small_printer,synth_small_nobg'.split(',')

metrics_names = ['Precision', 'Recall', 'F1-score', 'Accuracy']
out = edict()
conf_matrices = edict()
for dataset in datasets:
    out[dataset] = edict()
    conf_matrices[dataset] = edict()
    for model in models:
        val = data[dataset][model]
        
        preds, labels = get_class_preds(val.labels, val.preds, th=0.5, iou_th=0.5)
#         print(len(preds))
        out[dataset][model] = []
        out[dataset][model].append(metrics.precision_score(labels, preds, average='weighted'))
        out[dataset][model].append(metrics.recall_score(labels, preds, average='weighted'))
        out[dataset][model].append(metrics.f1_score(labels, preds, average='weighted'))
        out[dataset][model].append(metrics.accuracy_score(labels, preds))
        conf_matrices[dataset][model] = build_confusion_matrix(labels, preds, cfg.classes)

# %%
df_metrics = pd.DataFrame({(i,j): out[i][j] for i in datasets for j in models}, index=metrics_names)
df_metrics

# %%
def format_cm(cm):
    cm = copy(cm)
    l = cfg.abre_classes + ['Undet.', 'Total']
    cm.columns = l
    cm.index = l
    for i in range(len(cm)):
        new_vals = cm.iloc[i, :].to_numpy().astype(np.float64) / np.float64(cm.iloc[i, -1])
        cm.iloc[i, :] = pd.Series([val * 100 for val in new_vals], index = cm.index)
    cm.drop('Total', axis=0, inplace=True)
    cm.drop('Undet.', axis=0, inplace=True)
    cm.drop('Total', axis=1, inplace=True)
    cm = cm.round(1)
    return cm

format_cm(conf_matrices['real']['frcnn_default'])