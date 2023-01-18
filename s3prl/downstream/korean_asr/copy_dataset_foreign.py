# -*- coding: cp949 -*-
from glob import glob
import os, shutil
import soundfile as sf
import argparse
import numpy as np
import librosa
import soundfile as sf
import re
import json

parser = argparse.ArgumentParser(description='copy')
parser.add_argument('--start_dir', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/new_KoreanForeign')
parser.add_argument('--number', type=int, default=1)
parser.add_argument('--dest_dir', type=str, default='/Data~')
args = parser.parse_args()



def moving(dataset, data_dir):
    check = []
    for idx, data in enumerate(dataset):
        _dir, file_id = os.path.split(data)        
        input_txt, preprocess_txt = read_preprocess_text_file(data)
        #print('data {} input_txt {} preprocess_txt {} length {}'.format(data, input_txt, preprocess_txt, len(preprocess_txt)))
        
        
        if len(preprocess_txt) == 0:
            print('txt file is 0')
            check.append(data)
            if os.path.isfile(os.path.join(data_dir, file_id.replace('.txt', '.npy'))):
                print('file yes', os.path.join(data_dir, file_id.replace('.txt', '.npy')))
                print('data {} input_txt {} preprocess_txt {} length {}'.format(data, input_txt, preprocess_txt, len(preprocess_txt)))
                os.remove(os.path.join(data_dir, file_id.replace('.txt', '.npy')))
        else:
            with open(os.path.join(data_dir, file_id), 'w') as f:
                f.write(preprocess_txt)
                
    #print('check', check)

def bracket_filter(sentence):
    new_sentence = str()
    
    flag = False    
    i_idx = 0
    j_idx = 0

    for idx, ch in enumerate(sentence):        
        
        if ch == '(':
            i_idx = idx
            flag = True
        if ch == ')':
            j_idx = idx
            flag = False
        
        if flag is True:
            if 'SP:' in sentence[i_idx:i_idx+4]:
                continue
            elif 'NO:' in sentence[i_idx:i_idx+4]:
                if ch not in "NO:":
                    if ch == '(':
                        continue
                    new_sentence += ch #continue
            elif 'FP:' in sentence[i_idx:i_idx+4]:
                #if ch != 'F' or 'P' or ':':
                if ch not in "FP:":
                    if ch == '(':
                        continue
                    new_sentence += ch #continue
            
            elif 'SN:' in sentence[i_idx:i_idx+4]:
                #if ch != 'S' or 'N' or ':':
                if ch not in "SN:":
                    if ch == '(':
                        continue
                    new_sentence += ch #continue
            else:
                new_sentence += ch
            #elif 'NO' or 'FP' or 'SN' in sentence[i_idx:j_idx+1]:
                
        else:     
            if ch == ')':
                continue
            new_sentence += ch

    return new_sentence

def sentence_filter(raw_sentence):
    new_sentence = bracket_filter(raw_sentence)
    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    final_new_sentence = hangul.sub(' ', new_sentence)
    return final_new_sentence



def read_preprocess_text_file(file_path):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file) 
    raw_sentence = json_data['발화정보']['stt']
    #print(raw_sentence)
    
    '''
    with open(file_path, 'r') as f:
        raw_sentence = f.read()
    '''
    return raw_sentence, sentence_filter(raw_sentence)
    
'''
def read_preprocess_text_file(file_path):
    with open(file_path, 'r') as f:
        raw_sentence = f.read()
    return raw_sentence, sentence_filter(raw_sentence)
'''

train_path = os.path.join(args.start_dir, 'Training', 'label'+str(args.number))
#valid_path = os.path.join(args.start_dir, 'Validation', 'label'+str(args.number))

target_train_path = os.path.join(args.dest_dir, 'Training', 'label'+str(args.number))
if not os.path.exists(target_train_path):
    os.makedirs(target_train_path)
'''
target_valid_path = os.path.join(args.dest_dir, 'Validation', 'label'+str(args.number))
if not os.path.exists(target_valid_path):
    os.makedirs(target_valid_path)
'''

s_t_script = sorted(glob(train_path+'/*/*.json'))
#s_v_script = sorted(glob(valid_path+'/*/*.txt'))

#print('train script {} valid script {}'.format(len(s_t_script), len(s_v_script)))
print('train script {}'.format(len(s_t_script)))
moving(s_t_script, target_train_path)
#moving(s_v_script, target_valid_path)

'''
#previous
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
moving(s_t_script, target_train_path)
moving(s_v_script, target_valid_path)
'''
