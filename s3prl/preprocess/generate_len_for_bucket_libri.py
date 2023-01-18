# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ generate_len_for_bucket.py ]
#   Synopsis     [ preprocess audio speech to generate meta data for dataloader bucketing ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import pickle
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import soundfile as sf
import librosa

##################
# BOOLEAB STRING #
##################
def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-i', '--input_data', default='/home/kaen2891/workspace/data/LibriSpeech/', type=str, help='Path to your LibriSpeech directory', required=False)
    parser.add_argument('-o', '--output_path', default='./data/', type=str, help='Path to store output', required=False)
    parser.add_argument('-a', '--audio_extension', default='.wav', type=str, help='audio file type (.wav / .flac / .mp3 / etc)', required=False)
    parser.add_argument('-n', '--name', default='len_for_bucket', type=str, help='Name of the output directory', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)

    args = parser.parse_args()
    return args

# Resampling
def resampling(y, origin_sr, resample_sr):
    #y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, origin_sr, resample_sr)
    return resample

##################
# EXTRACT LENGTH #
##################
def extract_length(input_file, args):
    _, file_id = os.path.split(input_file)
    audio_extension = file_id.split('.')[-1].lower()
    
    if audio_extension == 'pcm':        
        with open(input_file, 'rb') as opened_pcm_file:
            buf = opened_pcm_file.read()
            pcm_data = np.frombuffer(buf, dtype = 'int16')
            wav_data = librosa.util.buf_to_float(pcm_data, 2) # (N,)
        wav_data = np.expand_dims(wav_data, -1)
        return wav_data.shape[0]
    else:        
        #_, based_sr = sf.read(input_file)
        based_sr = librosa.get_samplerate(input_file)
        if based_sr != 16000:
            y, _ = librosa.load(y=input_file, sr=based_sr)
            input_file = resampling(y, based_sr, 16000)
        wav, _ = torchaudio.load(input_file)
    return wav.size(-1) #(1, N)


###################
# GENERATE LENGTH #
###################
def generate_length(args, tr_set, audio_extension):
    
    for i, s in enumerate(tr_set):
        if os.path.isdir(os.path.join(args.input_data, s.lower())):
            s = s.lower()
            print('if, s', s)
        elif os.path.isdir(os.path.join(args.input_data, s.upper())):
            s = s.upper()
            print('else', s)
        else:
            print('all else')
            assert NotImplementedError

        print('')
        todo = list(Path(os.path.join(args.input_data, s)).rglob('*' + audio_extension)) # '*.pcm'
        print(f'Preprocessing data in: {s}, {len(todo)} audio files found.')

        output_dir = os.path.join(args.output_path, args.name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        print('Extracting audio length...', flush=True)
        tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_length)(str(file), args) for file in tqdm(todo))

        # sort by len
        sorted_todo = [os.path.join(s, str(todo[idx]).split(s+'/')[-1]) for idx in reversed(np.argsort(tr_x))]
        # Dump data
        df = pd.DataFrame(data={'file_path':[fp for fp in sorted_todo], 'length':list(reversed(sorted(tr_x))), 'label':None})
        df.to_csv(os.path.join(output_dir, tr_set[i] + '.csv'))

    print('All done, saved at', output_dir, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()
    print(args.input_data)
    
    #SETS = ['KoSpeech_1000hour', 'command_sentence_adult', 'command_sentence_child', 'command_sentence_elder', 'foreign_sentence', 'free_sentence_adult', 'free_sentence_child', 'free_sentence_elder'] # only use here
    # change the SETS list to match your dataset, for example:
    # SETS = ['train', 'dev', 'test']
    # SETS = ['TRAIN', 'TEST']
    SETS = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    
    # Select data sets
    for idx, s in enumerate(SETS):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [SETS[int(t)] for t in tr_set.split(' ')]
    print(tr_set)

    # Acoustic Feature Extraction & Make Data Table
    generate_length(args, tr_set, args.audio_extension)


if __name__ == '__main__':
    main()
