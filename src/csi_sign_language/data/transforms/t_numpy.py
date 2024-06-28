
import random 
import numpy as np
import copy
from .functional import rotate_and_crop, adjust_bright, to_gray, numpy2pil, pil2numpy
import torchvision.transforms.functional as F
from PIL import Image
import cv2
import numbers
import random

class Resize:
    def __init__(self, h, w) -> None:
        self.h = h
        self.w = w
        
    def __call__(self, video):
        modified = []
        for frame in video:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            modified.append(frame)
        return np.stack(modified)


class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        im_h, im_w, im_c = clip[0].shape
        if crop_w > im_w:
            pad = crop_w - im_w
            clip = [np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            w1 = 0
        else:
            w1 = random.randint(0, im_w - crop_w)

        if crop_h > im_h:
            pad = crop_h - im_h
            clip = [np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            h1 = 0
        else:
            h1 = random.randint(0, im_h - crop_h)

        return np.array([img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip])
        
        
class RandomHorizontalFlip(object):

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, video):
        #t h w c
        flag = random.random() < self.prob
        if flag:
            video = np.flip(video, axis=-2)
            video = np.ascontiguousarray(copy.deepcopy(video))
        return video

class RandomRotate(object):

    def __init__(self, prob, angle_range):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, video):
        #t, h, w, c
        flag = random.random() < self.prob
        if flag:
            T, H, W, C = video.shape
            angle =  random.uniform(self.angle_range[0], self.angle_range[1])
            video = [rotate_and_crop(frame, angle, 0, 0, H, W) for frame in video]
            video = np.array(video)
        return video


class RandomBrightJitter:
    def __init__(self, prob, factor_range) -> None:
        self.prob = prob
        self.factor_range = factor_range
    
    def __call__(self, video):
        #t, h, w, c
        flag = random.random() < self.prob
        if flag:
            factor = random.uniform(self.factor_range[0], self.factor_range[1])
            video = [adjust_bright(frame, factor) for frame in video]
            video = np.array(video)
        return video

class RandomGray:
    
    def __init__(self, prob) -> None:
        self.prob = prob
    
    def __call__(self, video):
        #t, h, w, c
        flag = random.random() < self.prob
        video = [to_gray(frame) for frame in video]
        video = np.array(video)
        return video