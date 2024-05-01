import os
from typing import Any
import numpy as np
import glob
import cv2 as cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchtext.vocab import vocab, build_vocab_from_iterator, Vocab
import torch.nn.functional as F
from einops import rearrange

from csi_sign_language.csi_typing import PaddingMode
from ...csi_typing import *
from ...utils.data import VideoGenerator, padding, load_vocab
from typing import *
from abc import ABC, abstractmethod

from ...utils.lmdb_tool import retrieve_data
import json
import yaml
import lmdb

class BasePhoenix14Dataset(Dataset, ABC):
    data_root: str
    def __init__(self, data_root, gloss_vocab_dir, type='train', multisigner=True, length_time=None, length_glosses=None,
                padding_mode : PaddingMode ='front'):
        if multisigner:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
            annotation_file = type + '.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-multisigner/features/fullFrame-210x260px', type)
        else:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
            annotation_file = type + '.SI5.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px', type)

        self._annotations = pd.read_csv(os.path.join(annotation_dir, annotation_file), delimiter='|')
        self._feature_dir = feature_dir
        self._data_root = data_root

        self._length_time = self.max_length_time if length_time == None else length_time
        self._length_gloss = self.max_length_gloss if length_glosses == None else length_glosses
        self._padding_mode = padding_mode

        self.gloss_vocab = load_vocab(gloss_vocab_dir)

    def __len__(self):
        return len(self._annotations)

    @abstractmethod
    def __getitem__(self, idx):
        return

    @property
    def max_length_time(self):
        max = 0
        for folder in self._annotations['folder']:
            file_list = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
            if len(file_list) >= max:
                max = len(file_list)
        return max

    @property
    def max_length_gloss(self):
        max = 0
        for glosses in self._annotations['annotation']:
            l = len(glosses.split())
            if l > max:
                max = l
        return max
    

class Phoenix14Dataset(BasePhoenix14Dataset):
    """
    Dataset for general RGB image with gloss label, the output is (frames, gloss_labels, frames_padding_mask, gloss_padding_mask)
    """
    def __init__(self, data_root, gloss_vocab_dir, type='train', multisigner=True, length_time=None, length_glosses=None,img_transform=None, transform=None):
        super().__init__(data_root, gloss_vocab_dir, type, multisigner, length_time, length_glosses, 'back')
        self._img_transform = img_transform
        self.transform = transform

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        anno = self._annotations['annotation'].iloc[idx]
        anno: List[str] = anno.split()
        anno: List[int] = self.gloss_vocab(anno)
        anno: np.ndarray = np.asarray(anno)

        folder: str = self._annotations['folder'].iloc[idx]
        frame_files: List[str] = self._get_frame_file_list_from_annotation(folder)

        video_gen: VideoGenerator = VideoGenerator(frame_files)
        frames: List[np.ndarray] = [frame if self._img_transform == None else self._img_transform(frame)  for frame in video_gen]
        # [t, h, w, c]
        frames: np.ndarray = np.stack(frames)

        # padding
        frames, frames_mask = padding(frames, 0, self._length_time, self._padding_mode)
        anno, anno_mask = padding(anno, 0, self._length_gloss, self._padding_mode)
        
        
        ret = dict(
            video=frames.astype('uint8'), #[t, h, w, c]
            annotation = anno, #[s]
            video_mask = frames_mask, #[t]
            annotation_mask = anno_mask #[s]
        )

        if self.transform:
            ret = self.transform(ret)
            
        return ret
    
    def _get_frame_file_list_from_annotation(self, folder: str) -> List[str]:
        """return frame file list with the frame order"""
        file_list: List[str] = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
        file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('-')[0][2:]))
        return file_list
        


class MyPhoenix14Dataset(Dataset):
    
    data_root: str

    
    def __init__(
        self, 
        data_root: str,
        subset: Union[Literal['multisigner'], Literal['si5']],
        mode: Union[Literal['train'], Literal['dev'], Literal['test']],
        gloss_length=None,
        video_length=None,
        transform=None) -> None:
        
        self.data_root = data_root
        self.subset = subset
        self.mode = mode
        self.subset_root = os.path.join(data_root, subset)
        with open(os.path.join(self.subset_root, 'info.json'), 'r') as f:
            self.info = json.load(f)
        # self.info = OmegaConf.load(os.path.join(self.subset_root, 'info.yaml'))
            
        self.vocab = self.create_vocab_from_list(self.info['vocab'])
        self.data_id: List[str] = self.info[mode]['data']

        self.transform = transform
        
        self.video_length = video_length
        self.gloss_length = gloss_length
        self.lmdb_env = None
    
    def get_stm(self):
        return os.path.join(self.subset_root, f'phoenix2014-groundtruth-{self.mode}.stm')
    
    def id2index(self, id):
        return self.data_id.index(id)
        
    def __getitem__(self, index) -> Any:
        if self.lmdb_env is None:
            self._init_db()
        
        id = self.data_id[index]
        data = retrieve_data(self.lmdb_env, id)
        video = data['video']
        video = rearrange(video ,'t c h w -> t h w c')
        gloss_label = data['gloss_labels']
        gloss = np.array(self.vocab(gloss_label), dtype='int64')
    
        ret = dict(
            id=id,
            video=video, #[t c h w], uint8, 0-255
            gloss=gloss,
            gloss_label=gloss_label)
        
        if self.transform is not None:
            ret = self.transform(ret)

        return ret
        
    def __len__(self):
        return len(self.data_id)
    
    def _init_db(self):
        self.lmdb_env = lmdb.open(
            os.path.join(self.data_root, self.subset, self.mode, 'feature_database'), 
            readonly=True,
            lock=False,
            create=False)
    
    def get_vocab(self):
        return self.vocab
    
    @staticmethod 
    def create_vocab_from_list(list: List[str]):
        return vocab(OrderedDict([(item, 1) for item in list]))

class CollateFn:
    
    def __init__(self, gloss_mode='padding', length_video=None, length_gloss=None) -> None:
        self.length_video = length_video
        self.length_gloss = length_gloss
        self.gloss_mode = gloss_mode
    
    def __call__(self, data) -> Any:
        #sort the data by video length in decreasing way for onxx
        data = sorted(data, key=lambda x: len(x['video']), reverse=True)
        
        video_batch = [item['video'] for item in data]
        gloss_batch = [item['gloss'] for item in data]
        gloss_label = [item['gloss_label'] for item in data]
        ids = [item['id'] for item in data]
        
        video, v_length = self._padding_temporal(video_batch, self.length_video)
        
        if self.gloss_mode=='concat':
            g_length = torch.tensor([len(item) for item in gloss_batch], dtype=torch.int32)
            gloss = torch.concat(gloss_batch)
        elif self.gloss_mode=='padding':
            gloss, g_length = self._padding_temporal(gloss_batch, self.length_gloss)
        
        video = rearrange(video, 'b t c h w -> b c t h w')
        
        return dict(
            id=ids,
            video=video,
            gloss=gloss,
            video_length=v_length,
            gloss_length=g_length,
            gloss_label=gloss_label
        )
    
    @staticmethod
    def _padding_temporal(batch_data: List[torch.Tensor], force_length=None):
        #[t, ....]
        if batch_data is not None:
            if not isinstance(batch_data[0], torch.Tensor):
                raise Exception('data in collate function must be a torch tensor!')
        if force_length is not None:
            #if force temporal length
            t_length = force_length
        else:
            #temporal should be the max
            t_length = max(data.size()[0] for data in batch_data)
        
        t_lengths_data = []
        ret_data = []
        for data in batch_data:
            t_length_data = data.size()[0]
            t_lengths_data.append(torch.tensor(t_length_data))
            delta_length = t_length - t_length_data
            assert delta_length >= 0
            if delta_length == 0:
                ret_data.append(data)
                continue
            
            data = torch.transpose(data, 0, -1)
            data = F.pad(data, (0, delta_length), mode='constant', value=0)
            data = torch.transpose(data, 0, -1)
            ret_data.append(data)
        
        return torch.stack(ret_data), torch.stack(t_lengths_data)
