# -*- coding: cp949 -*-
from glob import glob
import os, shutil
import soundfile as sf
import argparse
import numpy as np
import librosa
import soundfile as sf
import re

parser = argparse.ArgumentParser(description='copy')
parser.add_argument('--start_dir', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/asr_dataset/foreign_sentence/')
parser.add_argument('--current_dir', type=str, default='/Data/junewoo/feature/Kspon_NIKL_100E/')
parser.add_argument('--number', type=int, default=1)
parser.add_argument('--dest_dir', type=str, default='/Data~')
args = parser.parse_args()



def moving(dataset, data_dir, cur_dir):
    cnt = 0
    for idx, data in enumerate(dataset):
        _dir, file_id = os.path.split(data)
        wav_file = os.path.join(_dir, file_id.replace('.txt', '.wav'))
        cur_npy = os.path.join(cur_dir, file_id.replace('.txt', '.npy'))
        cur_txt = os.path.join(cur_dir, file_id)
        
        if os.path.isfile(cur_npy) and os.path.isfile(cur_txt):
            print('all True, cur_npy {} cur_txt {} check_cur_npy {} check_cur_txt {}'.format(cur_npy, cur_txt, os.path.isfile(cur_npy), os.path.isfile(cur_txt)))
            shutil.copy(wav_file, data_dir)            
            shutil.copy(cur_txt, data_dir)
            cnt += 1
    return cnt
            
        
        

train_path = os.path.join(args.start_dir, 'Training', str(args.number))
valid_path = os.path.join(args.start_dir, 'Validation', str(args.number))

target_train_path = os.path.join(args.dest_dir, 'Training', str(args.number))
if not os.path.exists(target_train_path):
    os.makedirs(target_train_path)
target_valid_path = os.path.join(args.dest_dir, 'Validation', str(args.number))
if not os.path.exists(target_valid_path):
    os.makedirs(target_valid_path)

s_t_script = sorted(glob(train_path+'/*/*.txt'))
s_v_script = sorted(glob(valid_path+'/*/*.txt'))

print('train script {} valid script {}'.format(len(s_t_script), len(s_v_script)))


c_train_path = os.path.join(args.current_dir, 'Training', str(args.number))
if not os.path.exists(c_train_path):
    os.makedirs(c_train_path)
c_valid_path = os.path.join(args.current_dir, 'Validation', str(args.number))
if not os.path.exists(c_valid_path):
    os.makedirs(c_valid_path)

train_cnt = moving(s_t_script, target_train_path, c_train_path)
valid_cnt = moving(s_v_script, target_valid_path, c_valid_path)
print('train_path {} valid_path {} data_number {} train_cnt {} valid_cnt {}'.format(target_train_path, target_valid_path, args.number, train_cnt, valid_cnt))