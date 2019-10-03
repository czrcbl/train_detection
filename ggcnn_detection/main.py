from __future__ import division, print_function, absolute_import
import numpy as np
import torch
import cv2
from skimage.filters import gaussian
import os
from os.path import join as pjoin
from .models.ggcnn import GGCNN


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
models_dir = pjoin(base_dir, 'data/grasping/models')


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img


def normalize(img):
    return np.clip((img - img.mean()), -1, 1)


class GraspOutput:
    """Utlity class to wrap the GDetector output."""
    def __init__(self, img, q_img, ang_img, width_img):
        self.img = img
        self.q_img = q_img
        self.ang_img = ang_img
        self.width_img = width_img


    def mask(self, bbox):
        """Return the an image of the grap image size, with zeros everywhere but inside the input bounding box."""
        mq_img, mang_img, mwidth_img = [bbox.mask_image(x) for x in [self.q_img, self.ang_img, self.width_img]]
        return mq_img, mang_img, mwidth_img
    
    def get_best(self, bbox=None):
        """Return the best draw on the region delimited by the input bounding box.
        
        Output
        ------
        (px, py) : Center point of the grasp
        bscore : Score of the grasp
        bdist : Distance to the grap point, extracted form the depth image
        bang : Angle of the grasp
        bwidth : Width of the grasp
        """
        if bbox is None:
            mq_img, mang_img, mwidth_img = self.q_img, self.ang_img, self.width_img
        else:
            mq_img, mang_img, mwidth_img = self.mask(bbox)

        ind = np.argmax(mq_img)
        px = ind % mq_img.shape[1]
        py = ind // mq_img.shape[1]
        bscore = mq_img[py, px]
        bang = mang_img[py, px]
        bwidth = mwidth_img[py, px]
        bdist = self.img[px, py]

        return (px, py), bscore, bdist, bang, bwidth

    def draw_best(self, img, bbox=None):

        img = np.copy(img)
        p, bscore, bdist, bang, bwidth = self.get_best(bbox=bbox)
        height, width = img.shape[:2]
        thickness = 1 + int(img.shape[1]/300)
        color = (255, 0, 0)
        axis = (int(bwidth), int(height/40))
        cv2.ellipse(img, p, axis, bang * 180 / np.pi, 0.0, 360.0, color, thickness)
        # text = str(bdist)
        # font_scale = 0.5/600 * width
        # thickness = int(2/600 * width)
        # cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return img


class GDetector:
    """Class to wrap the GGCNN in order to perform predictions"""
    def __init__(self, models_dir=models_dir, depth=True, rgb=False, device='cpu'):

        if rgb or (not depth):
            raise ValueError('rgb not implemented.')

        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'gpu':
            self.device = torch.device('cuda')
        else:
            raise ValueError('Invalid device {}.'.format(device))
        
        net = GGCNN()
        net.load_state_dict(torch.load(pjoin(models_dir, 'ggcnn_model.pt')))
        self.net = net.to(self.device)
        self.net.eval()


    def detect(self, img):
        """Transforms and detect the grasps on the input depth image."""
        n_img = normalize(img)
        t_img = torch.tensor(n_img)[None, None, :, :].float().to(self.device)

        with torch.no_grad():
            pos_output, cos_output, sin_output, width_output = self.net(t_img)
    
        q_img, ang_img, width_img = post_process_output(pos_output, cos_output, sin_output, width_output)

        return GraspOutput(img, q_img, ang_img, width_img)