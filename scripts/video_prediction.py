import _fix_path
import argparse
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd
import time
from os.path import join as pjoin

from detection import Detector
from traindet import config as cfg


def predict_frames(in_fn, out_fn, det, th=0.5):

    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)

    cap = cv2.VideoCapture(in_fn)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(out_fn, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    i = 0
    while True:
        tic = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8')
        bboxes, _ = det.detect(frame, threshold=th)
        dimg = bboxes.draw(frame)
        
        img = cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR)
        out.write(img) 
        print(f'Frame {i} processed, elapsed time: {time.time() - tic:.3f}s.')
        i += 1
        
    cap.release()
    out.release()

def parse_args():
    parser = argparse.ArgumentParser('Make predictions on every frame of input video and save output.')
    parser.add_argument('--input-file', help='Path to the input video.')
    parser.add_argument('--output-path', help='Path to the output.', default='')
    parser.add_argument('--model', help='Model to use.')
    parser.add_argument('--dataset', help='Dataset in which the model was trained.')
    parser.add_argument('--th', default=0.5, help='Detection threshold to use.')
    parser.add_argument('--ctx', default='gpu', help='Context (cpu or gpu), default: gpu.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    det = Detector(model=args.model, dataset=args.dataset, ctx=args.ctx)
    if args.output_path:
        out_file = args.output_path
    else:
        out_file = pjoin(cfg.data_folder, 'videos', f'{args.model}_{args.dataset}_{args.th}.mp4')
    predict_frames(args.input_file, out_file, det, th=args.th)
    


if __name__ == '__main__':

    main()






