import click
from functools import partial
import os
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import sys
from multiprocessing import Pool
sys.path.append('src')

from tqdm import tqdm
import glob
from typing import *
import numpy as np
import cv2
from einops import rearrange
from csi_sign_language.utils.data import VideoGenerator
from omegaconf import OmegaConf
import json
from csi_sign_language.utils.lmdb_tool import store_data, retrieve_data
import lmdb
import shutil

#preprocess the data into lmdb format

@click.command()
@click.option('--data_root', default='dataset/phoenix2014-release')
@click.option('--output_root', default='preprocessed/ph14')
@click.option('--frame_size', nargs=2, default=(256, 256))
@click.option('--subset', default='multisigner')
@click.option('--multiprocess', default=True)
@click.option('--chunk_size', default=2, help='the chunk size of data submitted to each process to handle')
@click.option('--num_p', default=10, help="number of process to create")
@click.option('--specials', default=['<blank>'], multiple=True)
def main(data_root, output_root, frame_size, subset, multiprocess, chunk_size, num_p, specials):
    
    vocab, vocab_SI5 = generate_vocab(data_root, output_root, specials)
    info = OmegaConf.create()
    info.author = 'jingyan wang'
    info.email = '2533494w@student.gla.ac.uk'
    
    subset_root = os.path.join(output_root, subset)
    os.makedirs(subset_root, exist_ok=True)
    
    #put ground truth to the root
    if subset == 'multisigner':
        data_root_subset = os.path.join(data_root, 'phoenix-2014-multisigner')
    if subset == 'si5':
        data_root_subset = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5')

    shutil.copy(os.path.join(data_root_subset, 'evaluation/phoenix2014-groundtruth-dev.stm'), subset_root)
    shutil.copy(os.path.join(data_root_subset, 'evaluation/phoenix2014-groundtruth-test.stm'), subset_root)
    multi_signer = True if subset == 'multisigner' else False
    v = vocab if subset == 'multisigner' else vocab_SI5
    info['vocab'] = v.get_itos()

    for mode in ('train', 'dev', 'test'):
        annotations, feature_dir, max_lgt_v, max_lgt_g= get_basic_info(data_root, mode, multisigner=multi_signer)
        data_length = len(annotations)
        print(f'creating {subset}-{mode}')
        info[mode] = {}
        info[mode]['max_length_video'] = max_lgt_v
        info[mode]['max_length_gloss'] = max_lgt_g
        os.makedirs(os.path.join(subset_root, mode), exist_ok=True)
        if multiprocess:
            keys = run_mp_cmd(partial(process_data, mode, v, frame_size, annotations, feature_dir, subset, subset_root), range(data_length), num_p=num_p,chunk_size=chunk_size)
        else:
            keys = process_data(list(range(data_length)), info, mode, v, frame_size, annotations, feature_dir, subset, subset_root)
        
        info[mode]['data'] = keys

    info_d = OmegaConf.to_container(info)
    with open(os.path.join(subset_root, 'info.json'), 'w') as f:
        json.dump(info_d, f, indent=4)
            
def process_data(mode, v, frame_size, annotations, feature_dir, subset, subset_root, idxes):
    env = lmdb.open(os.path.join(subset_root, mode,'feature_database'), map_size=1099511627776)
    keys = []
    for idx in idxes:
        id, video, signer, gloss_labels = get_single_data(idx, annotations, v, feature_dir, frame_size)
        data = dict(
            id=id,
            video=video,
            signer=signer,
            gloss_labels=gloss_labels
        )
        store_data(env, id, data)
        keys.append(id)
    env.close()
    return keys

def run_mp_cmd(func, data_indexes, num_p, chunk_size):
    process_args = [data_indexes[i:i + chunk_size] for i in range(0, len(data_indexes), chunk_size)]  
    with Pool(num_p) as p:
        keys = []
        for result in tqdm(p.imap(func, process_args), total=len(process_args)):
            keys += result 
    return keys


def get_single_data(idx, annotations, gloss_vocab, feature_dir, frame_size=(224, 224)):
    anno_str: List[str] = annotations['annotation'].iloc[idx].split()
    id: str = annotations['id'][idx]
    signer: str = annotations['signer'][idx]


    folder: str = annotations['folder'].iloc[idx]
    frame_files: List[str] = get_frame_file_list_from_annotation(feature_dir, folder)

    video_gen: VideoGenerator = VideoGenerator(frame_files)
    frames: List[np.ndarray] = [cv2.cvtColor(cv2.resize(frame, frame_size, cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) for frame in video_gen]
    # [t, h, w, c]
    frames: np.ndarray = np.stack(frames)

    frames = rearrange(frames, 't h w c -> t c h w')
    
    
    return id, frames, signer, anno_str
    
    
def get_basic_info(data_root, type='train', multisigner=True,):
    
    if multisigner:
        annotation_dir = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
        annotation_file = type + '.corpus.csv'
        feature_dir = os.path.join('phoenix-2014-multisigner/features/fullFrame-210x260px', type)
    else:
        annotation_dir = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
        annotation_file = type + '.SI5.corpus.csv'
        feature_dir = os.path.join('phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px', type)

    annotations = pd.read_csv(os.path.join(annotation_dir, annotation_file), delimiter='|')
    feature_dir = os.path.join(data_root, feature_dir)

    max_lgt_vid = max_length_time(annotations, feature_dir)
    max_lgt_gloss = max_length_gloss(annotations)
    
    return annotations, feature_dir, max_lgt_vid, max_lgt_gloss

def get_frame_file_list_from_annotation(feature_dir, folder: str) -> List[str]:
    """return frame file list with the frame order"""
    file_list: List[str] = glob.glob(os.path.join(feature_dir, folder))
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('-')[0][2:]))
    return file_list


def max_length_time(annotations, feature_dir):
    max = 0
    for folder in annotations['folder']:
        file_list = glob.glob(os.path.join(feature_dir, folder))
        if len(file_list) >= max:
            max = len(file_list)
    return max

def max_length_gloss(annotations):
    max = 0
    for glosses in annotations['annotation']:
        data_indexes = len(glosses.split())
        if data_indexes > max:
            max = data_indexes
    return max

def create_glossdictionary(annotations, specials):
    def tokens():
        for annotation in annotations['annotation']:
            yield annotation.split()
    vocab = build_vocab_from_iterator(tokens(), special_first=True, specials=specials)
    return vocab
    
def generate_vocab(data_root, output_root, specials):
    print(os.getcwd())
    annotation_file_multi = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
    annotation_file_single = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
    
    multi = []
    si5 = []
    for type in ('dev', 'train', 'test'):
        multi.append(pd.read_csv(os.path.join(annotation_file_multi, type+'.corpus.csv'), delimiter='|'))
        si5.append(pd.read_csv(os.path.join(annotation_file_single, type+'.SI5.corpus.csv'), delimiter='|'))
    
    vocab_multi = create_glossdictionary(pd.concat(multi), specials)
    vocab_single =  create_glossdictionary(pd.concat(si5), specials)
    
    return vocab_multi, vocab_single


if __name__ == '__main__':
    main()