import unittest

from traindet import config as cfg
from traindet.train_utils import get_dataset

# from traindet.utils import RealDataset, 

class TestDatasets(unittest.TestCase):

    def test_split(self):
        
        for dataset_name in cfg.dataset_names:
            if 'mixed' not in dataset_name:
                trn_ds, val_ds, _ = get_dataset(dataset_name)
                trn, val = len(trn_ds), len(val_ds)
                self.assertAlmostEqual(trn/(trn + val), 0.7, places=2)

    
    def test_train_test_different(self):

        for dataset_name in cfg.dataset_names:
            trn_ds, val_ds, _ = get_dataset(dataset_name)
            for fn1 in trn_ds.fns:
                for fn2 in val_ds.fns:
                    self.assertNotEqual(fn1, fn2)


if __name__ == '__main__':

    unittest.main()