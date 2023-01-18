
#-*- coding: utf-8 -*-

import os
import sys
import math
import time
import torch
import random
import six
import threading
import logging
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np
from warnings import warn
from numpy import ma
from torch.utils.data.sampler import Sampler

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

random_seed = 2891

import torch
import torch.nn as nn
import torch.optim as optim
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

from functools import lru_cache

PAD = 0
scene_length = None

def use_scene_length(input_window_length):
    global scene_length
    scene_length = input_window_length
    return scene_length

np.seterr(divide = 'ignore')
np.seterr(divide = 'warn')

def preemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0:
        return signal
    signal, _ = librosa.effects.trim(signal)
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ Mean and variance normalization """
    
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    
    std_dev = np.std(audio_feature, axis=axis) + 1e-9
    
    normalized = (audio_feature - mean) / std_dev
    '''
    mean = np.mean(audio_feature)
    std_dev = np.std(audio_feature) + 1e-9
    normalized = (audio_feature - mean) / std_dev
    '''
    return normalized


class SpectrogramDataset(Dataset):
    def __init__(self, config, data_list, char2index, sos_id, eos_id, data_dir, target_dict):
        super(SpectrogramDataset, self).__init__()
        """
        Dataset loads data from a list contatining wav_name, transcripts, speaker_id by dictionary.
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds.
        :param data_list: List of dictionary. key : "wav", "text", "speaker_id"
        :param char2index: Dictionary mapping character to index value.
        :param sos_id: Start token index.
        :param eos_id: End token index.
        :param normalize: Normalized by instance-wise standardazation.
        """
        self.audio_conf = config
        self.wav_list = data_list[0]
        self.script_list = data_list[1]
        self.size = len(self.wav_list)
        self.char2index = char2index
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.PAD = 0
        self.normalize = config['norm']
        self.data_dir = data_dir
        self.target_dict = target_dict
        
        self.min_level_db = -100

    def __getitem__(self, index):
        audio_path = self.wav_list[index]
        
        transcript = self.script_list[index]
        spect = self.parse_audio(audio_path, self.data_dir)
        transcript = self.parse_transcript(transcript)
        
        return spect, transcript

    def parse_audio(self, audio_path, data_dir):
        _, file_names = os.path.split(audio_path)
        
        feature = np.load(os.path.join(data_dir, file_names.replace('.pcm','.npy')))        
        
        if self.normalize == 0:
        
            feature = 20 * np.log10(np.maximum(1e-5, feature))
            feature = np.clip(-(feature - self.min_level_db) / self.min_level_db, 0, 1)
        elif self.normalize == 1:
            
            mel_dim = feature.shape[0]
            
            for k in range(mel_dim):
                dim = feature[k]
                dim_mean = dim.mean()
                normalize_dim = (dim+1e-9) / (dim_mean + 1e-9)
                feature[k] = normalize_dim
        
        elif self.normalize == 2:
            mean = np.mean(feature)
            std = np.std(feature)
            feature -= mean
            feature /= std
        
        elif self.normalize == 3:
            mel_dim = feature.shape[0]
            
            for k in range(mel_dim):
                dim = feature[k]
                dim_mean = np.mean(dim)
                dim_std = np.std(dim)
                feature[k] -= dim_mean
                feature[k] /= dim_std
        else:
            pass
        
        return torch.FloatTensor(feature)
        
    def parse_transcript(self, transcript):
        key = transcript.split('/')[-1].split('.')[0]
        script = self.target_dict[key]
        
        tokens = script.split(' ')
        result = list()
        result.append(self.sos_id)
        for i in range(len(tokens)):
            if len(tokens[i]) > 0:
                result.append(int(tokens[i]))
        result.append(self.eos_id)
        
        return result

    def __len__(self):
        return self.size


def _collate_fn(batch):

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    seq_lengths    = [s[0].size(1) for s in batch]
    target_lengths = [len(s[1]) for s in batch]
    
    seq_size = max(seq_lengths)
    scene_length = 32
    _, remain = divmod(seq_size, scene_length)
    mod = scene_length - remain
    max_seq_size = seq_size + mod
    
    max_target_size = max(target_lengths)

    feat_size = batch[0][0].size(0)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets

    
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
