
import random 
import numpy as np
import copy
from .functional import rotate_and_crop, adjust_bright, to_gray, numpy2pil, pil2numpy
import torchvision.transforms.functional as F
from PIL import Image
import cv2

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