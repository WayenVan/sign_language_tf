import numpy as np
import cv2
import typing
from ..csi_typing import PaddingMode
from collections import OrderedDict
from torchtext.vocab import vocab

def mapping_0_1(color_range, t):
    """mapping color from other range to 0., 1.
    """
    return (t - color_range[0]) / (color_range[1] - color_range[0])


def load_vocab(file_dir):
    with open(file_dir, 'r') as f:
        _dict = OrderedDict([(item.strip(), 1) for item in f])
    return vocab(_dict)

def list2vocab(l):
    _dict = OrderedDict([(k, 1) for k in l])
    return vocab(_dict)

class VideoGenerator:

    def __init__(self, frame_list: typing.List[str]):
        self.__frame_list = frame_list

    def __iter__(self) -> np.ndarray:
        for file in self.__frame_list:
            yield cv2.imread(file)
            
def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

def padding(data: np.ndarray, axis: int, length: int, padding_mode: PaddingMode):

    delta_legnth = length - data.shape[axis]
    if delta_legnth == 0:
        return data, np.ma.make_mask(np.ones(length))
    assert delta_legnth > 0
        

    npad = [[0, 0] for i in data.shape]
    if padding_mode == 'front':
        npad[axis][0] = delta_legnth
        mask = np.ones(length)
        mask[:delta_legnth] = 0
    elif padding_mode == 'back':
        npad[axis][1] = delta_legnth
        mask = np.ones(length)
        mask[-delta_legnth:] = 0
    else:
        raise Exception('padding_mode should be front or back')
    return np.pad(data, npad, mode='constant', constant_values=0), np.ma.make_mask(mask)

def stand(x, axis):
    mean = np.mean(x, axis, keepdims=True)
    std = np.std(x, axis, keepdims=True)
    return (x - mean)/ (std + 1e-7)

def norm(x, axis):
    mmax = np.max(x, axis, keepdims=True)
    mmin = np.min(x, axis, keepdims=True)
    return (x - mmin) / (mmax - mmin + 1e-7)

def interp(x: np.array, mask=None):
    """interp the signal of each channel in time
    :param x: [time, nodes, channel]
    :param mask: mask in time
    """
    num_nodes = x.shape[-2]
    channels = x.shape[-1]
    for node_idx in range(num_nodes):
        for channel_idx in range(channels):
            signal = x[:, node_idx, channel_idx] 
            no_zero_indices = np.where(signal != 0)[0]
            zero_indices = np.where(signal == 0)[0]
            if len(no_zero_indices) == 0:
                interp_values = signal[zero_indices]
            else:
                interp_values = np.interp(zero_indices, no_zero_indices, signal[no_zero_indices])
            x[zero_indices, node_idx, channel_idx] = interp_values
    
    if mask is not None:
        x[~mask, :, :] = 0
        
    return x
