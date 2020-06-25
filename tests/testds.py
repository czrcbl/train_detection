import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
if path not in sys.path:
    sys.path.insert(0, path)
import unittest
from traindet import config as cfg
from traindet.utils import SpecRealDataset, SpecSynthDataset
import matplotlib.pyplot as plt

ds = SpecRealDataset(tclass='part2', mode='all')
print(len(ds))
print(ds[0])
# ds = SpecRealDataset(tclass='part2')
# ds = SpecSynthDataset(root=os.path.join(cfg.dataset_folder, 'synth_small_printer'), tclass='part2')

# img, tgt = ds[0]
# print(tgt)
# plt.imshow(img.asnumpy().astype('uint8'))
# plt.show()