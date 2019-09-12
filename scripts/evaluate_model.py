import os
import argparse

from traindet.utils import load_map

def parse_args():

    parser = argparse.ArgumentParser('Get a report on the performance of the \
        input model on the input dataset.')
    parser.add_argument('--model')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    map_score, epoch = load_map(args.model, args.dataset)

    print(map_score)
    print(f'\nMax map reached on epoch:\n')
    print(epoch)

if __name__ == '__main__':
    main()

