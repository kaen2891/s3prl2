"""
Copyright 2019-present NAVER Corp.
"""

#-*- coding: utf-8 -*-

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue

            tmp = line.strip().split('\t')
            #print('temp = ', tmp)
            index = tmp[0]
            try:
                char = tmp[1]
            except:
                char = ' '
            
            #index, char = line.strip().split('\t')
            #index, char, freq = line.strip().split('\t')
            char = char.strip()
            #if len(char) == 0:
            #                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char
