import _fix_path
import argparse
import pickle
import os
from os.path import join as pjoin

from traindet import get_predictions
from traindet.train_utils import get_dataset
from traindet.utils import load_model
from traindet import config as cfg


def parse_args():

    parser = argparse.ArgumentParser(description='Run the trained model on the validation set.')
    parser.add_argument('--model-type', help='Model type.')
    parser.add_argument('--model', default='', help='Model to use.')
    parser.add_argument('--dataset', default='', help='Dataset to use.')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    dataset = args.dataset
    model = args.model
    
    _, val_ds, _ = get_dataset(dataset)
    print(f'Prediction {dataset} dataset with {model} model.')
    net, trans = load_model(args.model_type, model, dataset)
    preds, labels = get_predictions(net, val_ds, trans)
    folder = pjoin(cfg.gen_data_folder, 'predictions')
    try:
        os.makedirs(folder)
    except OSError:
        pass
    with open(pjoin(folder, f'{model}_{dataset}.pkl'), 'wb') as f:
        pickle.dump({'preds': preds, 'labels': labels}, f)


if __name__ == '__main__':
    main()