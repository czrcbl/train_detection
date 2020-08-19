# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
import sys
sys.path.insert(0, '../')


# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from traindet.utils import load_predictions


# %%
data = load_predictions('ssd_default,yolo_default,frcnn_default','real,synth_small_printer,synth_small_nobg')


# %%
preds = data['synth_small_printer']['frcnn_default']['preds']
labels = data['synth_small_printer']['frcnn_default']['labels']


# %%
def iou(b0, b1):
    
    w0 = b0[2] - b0[0]
    h0 = b0[3] - b0[1]
    A0 = w0 * h0
    w1 = b1[2] - b1[0]
    h1 = b1[3] - b1[1]
    A1 = w1 * h1
    
    inter = (min(b0[2], b1[2]) - max(b0[0], b1[0])) * (min(b0[3], b1[3]) - max(b0[1], b1[1]))
    
    IOU = inter / (A0 + A1 - inter)
    
    return IOU


# %%
iou([0,0,100,100], [0, 0, 100, 100]), iou([0,0,50,50], [0, 0, 100, 100])


# %%
iou_th = 0.5


# %%
resultsd = defaultdict(lambda: {'results': [], 'scores': [], 'gt': 0})
scores = []

for p, l in zip(preds, labels):
    used_bbs = []
    for pbb in p:
        if int(pbb[-1]) == -1:
            continue
        right = False
        for ln, lbb in enumerate(l):
            if (int(pbb[4]) == int(lbb[4])) and (iou(pbb, lbb) > iou_th):
                if (ln not in used_bbs):
                    used_bbs.append(ln)
                    right = True
                    
        resultsd[int(pbb[4])]['results'].append(1 if right else 0)
        resultsd[int(pbb[4])]['scores'].append(pbb[-1])

for l in labels:
    for lbb in l:
        resultsd[int(lbb[4])]['gt'] += 1


# %%
nresults = {}
for _id, data in resultsd.items():
    new_data = {}
    args = [x for x, y in sorted(enumerate(data['scores']), key=lambda a:a[1], reverse=True)]
    new_data['scores'] = [data['scores'][i] for i in args]
    new_data['results'] = [data['results'][i] for i in args]
    new_data['gt'] = data['gt']
    nresults[_id] = new_data


# %%
def calc_curves(scores, results, gt):
    scores = np.array(scores)
    results = np.array(results)
    precision = []
    recall = []
    for i in range(len(scores)):
        vals = results[:i + 1]
        TP = vals.sum()
        FP = len(vals) - vals.sum()
        precision.append(TP / (TP + FP))
        recall.append(TP / gt)
    
#     recall.insert(0, 0)
#     precision.insert(0, 1)
    return recall, precision
    


# %%
recall, precision = calc_curves(nresults[0]['scores'], nresults[0]['results'], nresults[0]['gt'])


# %%
from sklearn.metrics import auc


# %%
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
part_names = ['Tray', 'Dosing Nozzle', 'Button Pad', 'Part 01', 'Part 02', 'Part 03']
for cls_id, ax, name in zip(nresults.keys(), axs.flat, part_names):
    scores = nresults[cls_id]['scores']
    results = nresults[cls_id]['results']
    gt = nresults[cls_id]['gt']
    recall, precision = calc_curves(scores, results, gt)
    A = auc(recall, precision)
    ax.step(recall, precision, where='mid', label=f'AUC = {A:.2f}')
    ax.fill_between(recall, 0, precision, alpha=.3)
    # ax.set(xlim=(0, len(x) - 1), ylim=(0, None), xticks=x)
    ax.grid(True)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(name)
    ax.legend()
    ax.set_ylim([0, 1.05])


# %%
# fig.savefig('precision_recall_curves_synth_trans.pdf', bbox_inches='tight')


