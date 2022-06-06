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

class SpectrogramDataset(Dataset):
    def __init__(self, wav_list):
        super(SpectrogramDataset, self).__init__()
        
        self.wav_list = wav_list
        self.size = len(self.wav_list)

    def __getitem__(self, index):
        audio_path = self.wav_list[index]        
        spect, name = self.parse_audio(audio_path)        
        return spect, name

    def parse_audio(self, audio_path):
        file_dir, audio_extension = audio_path.split('.')
        _, file_name = os.path.split(file_dir)
        
        if audio_extension.lower() == 'wav':
            wav_data, _ = librosa.load(audio_path, 16000)        
        else:        
            with open(audio_path, 'rb') as opened_pcm_file:
                buf = opened_pcm_file.read()
                pcm_data = np.frombuffer(buf, dtype = 'int16')
                wav_data = librosa.util.buf_to_float(pcm_data, 2)
        
        return torch.FloatTensor(wav_data), file_name
        #return np.expand_dims(wav_data, axis=0), file_name
        
    
    def __len__(self):
        return self.size

# just only one batch

def _collate_fn(batch):
    #print('batch', batch)
    batch = sorted(batch, key=lambda sample: sample[0].size(-1), reverse=True)
    #print('batch', batch)
    seq_lengths    = [s[0].size(-1) for s in batch]
    #print('seq_lengths', seq_lengths)
    
    max_seq_size = max(seq_lengths)
        
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size)
    targets = list()
    
    for x in range(batch_size):        
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        
        seqs[x].narrow(0, 0, len(tensor)).copy_(tensor)
        targets.append(target)

    seq_lengths = torch.IntTensor(seq_lengths)    
    return seqs.squeeze(), targets


    
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

