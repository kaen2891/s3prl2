import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_sinusoid_encoding_table
import random

import copy
import math
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):   
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.attn_score = self.self_attn(src, src, src, attn_mask=src_mask,
                                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
       
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                      tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, self.attn_score = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                                                        key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, self.enc_dec_attn_score = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                       num_decoder_layers=6, enc_feedforward=2048, dec_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, enc_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dec_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                      src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class DownstreamConfig(object):
    """Configuration class to store the configuration of a `ASR_Model` for downstream.
    """
    def __init__(self, config):
        self.device = config['device']        
        self.d_model = int(config['hidden_size'])        
        self.num_enc_layers = int(config['num_enc_layers'])
        self.num_dec_layers = int(config['num_dec_layers'])
        self.num_attention_heads = int(config['num_attention_heads'])                
        self.enc_intermediate_size = int(config['enc_intermediate_size'])
        self.dec_intermediate_size = int(config['dec_intermediate_size'])
        self.transformer_dropout_prob = float(config['transformer_dropout_prob'])
        
        self.num_rnn_layers = int(config['rnn_num_layers'])
        self.conv_dropout_prob = float(config['conv_dropout_prob'])
        self.max_spec_length = int(config['max_spec_length'])
        self.max_text_length = int(config['max_text_length'])        
        self.isPos = bool(config['position_encoding'])
        self.isBidirectional = bool(config['bidirectional'])
        self.isSubsample = bool(config['subsampling'])


class Model(nn.Module):
    def __init__(self, vocab_len, sos_id, eos_id, d_model=512, nhead=8, num_encoder_layers=6, 
                       num_decoder_layers=6, max_seq_len=None, enc_feedforward=2048, dec_feedforward=2048,
                       dropout=0.1, max_length=65, padding_idx=0, mask_idx=0, device=None, 
                       num_lstm=6, bi=False, sub=False, pos=False, training=False):
        super(Model, self).__init__()
        self.device = device
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.max_seq_len = max_seq_len
        self.max_length = max_length
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.num_lstm = num_lstm
        self.bidirectional = bi
        self.subsampling = sub
        self.isPos = pos
        self.training = training
            
        self.conv = nn.Sequential( #input: 786, T
            nn.Conv2d(1, 16, kernel_size=(10, 3), stride=(3, 1), padding=(0, 1)), #254
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(3, 1), padding=(0, 1)), #84
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), #42
            nn.Dropout(0.3),

            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(1, 1), padding=(0, 1)), #38
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=(1, 1), padding=(0, 1)), #35
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=(1, 1), padding=(0, 1)), #32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), #6 16, 28
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 1)), #4 14, 26
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 1)), #2 12, 24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # 
        )
        
        self.fc_hidden1 = 512
        self.CNN_embed_dim = 256
        
        self.fc1 = nn.Linear(64 * 5 * 4, self.fc_hidden1)   
        self.fc2 = nn.Linear(self.fc_hidden1, self.CNN_embed_dim)
        #self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables

        print('self sub', self.subsampling)
        print('self pos', self.isPos)
        self.sub_linear = nn.Linear(self.d_model, self.d_model//2) # 256-->128
                        
        if self.num_lstm >= 1:
            if self.bidirectional:                
                if self.subsampling:                    
                    self.maxpool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
                    self.avgpool = nn.AvgPool2d(kernel_size=3, padding=1, stride=(1,2))
                    self.LSTM = nn.LSTM(self.d_model//2, self.d_model//2, self.num_lstm, batch_first=True, bidirectional=True)
                    print('in sub')         
                else:
                    print('no sub', self.d_model//2)
                    self.LSTM = nn.LSTM(self.d_model//2, self.d_model//2, self.num_lstm, batch_first=True, bidirectional=True) # num_layer=3
            else:
                self.LSTM = nn.LSTM(self.d_model, self.d_model, self.num_lstm, batch_first=True, bidirectional=False) # num_layer=3
        
        
        self.transformer = Transformer(d_model=self.d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, enc_feedforward=enc_feedforward, 
                                       dec_feedforward=dec_feedforward, dropout=dropout)

        self.embedding = nn.Embedding(vocab_len, self.d_model, padding_idx=self.padding_idx)
        self.enc_pos_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_seq_len+1, self.d_model, 
                                                                    padding_idx=self.padding_idx), freeze=True)
        self.dec_pos_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_length+1, self.d_model, 
                                                                    padding_idx=self.padding_idx), freeze=True)

        self.classifier = nn.Linear(self.d_model, vocab_len)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    
    def forward(self, src, tgt, hop_num=None, slice_num=None, sub_mode=None, spec_length=None, src_mask=None, tgt_mask=None, memory_mask=None,
                      src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, mode=None, mask=None):
        # B, 1, D, T (21, 1, 768, 3008)
        # spec_length 3008, hop_num 8, iter_num 376, iter_convlstm_num 372
        iter_num = spec_length // hop_num
        
        cnn_embed_seq = list()
        
        if hop_num == 8:
            iter_convlstm_num = iter_num-4
        elif hop_num == 16:
            iter_convlstm_num = iter_num-1
        elif hop_num == 4:
            iter_convlstm_num = iter_num-7
        else:
            print('conv lstm error, check please please')
        
        #print(f'input src {src.size()} tgt {tgt.size()} iter_num {iter_num}, spec_length {spec_length}, hop_num {hop_num} iter_convlstm_num {iter_convlstm_num}')
        
        for i in range(iter_convlstm_num):
            t = i*hop_num #hop_num 8
            x_t = src[:, :, :, t:t+slice_num] #slice_num = 32 --> 0 ~ 32 --> shape 21.1.768.32 
            #print(f'x_t {x_t.size()}')
            x_t = self.conv(x_t) # --> shape 21, 64, 5, 4 
            #print(f'after conv, x_t {x_t.size()}')
            x_t = x_t.view(x_t.size(0), -1)           # flatten the output of conv # --> shape 21, 1280
            #print(f'after view, x_t {x_t.size()}')
            x_t = F.relu(self.fc1(x_t))
            x_t = F.dropout(x_t, p=0.3, training=self.training)
            if self.d_model != 512:
                x_t = F.relu(self.fc2(x_t))
            #print(f'after fc2, x_t {x_t.size()}') # 21, 256
            
            cnn_embed_seq.append(x_t)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1) # B, T, 256
        #print(f'after cnn_embed seq, cnn_embed_seq {cnn_embed_seq.size()}') # 21, 372, 256
        if self.num_lstm >= 1:            
            if self.bidirectional and self.subsampling:
                if sub_mode == 'max':
                    cnn_embed_seq = self.maxpool(cnn_embed_seq.unsqueeze(1))
                    cnn_embed_seq = cnn_embed_seq.squeeze()
                elif sub_mode == 'avg':
                    cnn_embed_seq = self.avgpool(cnn_embed_seq.unsqueeze(1))
                    cnn_embed_seq = cnn_embed_seq.squeeze()
                elif sub_mode == 'linear':
                    cnn_embed_seq = self.sub_linear(cnn_embed_seq)
                
            src, _ = self.LSTM(cnn_embed_seq)
        else:
            src = cnn_embed_seq
        
        #print(f'after rnn, src = {src.size()}') # 21, 372, 256
        # add pos
        if self.isPos:
            src_pos = torch.LongTensor(range(src.size(1))).to(self.device)
            src = src + self.enc_pos_enc(src_pos)
            
        if mode == 'train':
            # Enhance Decoder's representation
            #if mask == None:
            
            for i in range(len(tgt)):
                for j in range(len(tgt[i])):
                    if tgt[i][j] == 0:
                        break
                    
                    dice = random.random()
                    if dice <= mask:
                        tgt[i][j] = self.mask_idx
            
            tgt_key_padding_mask = (tgt == self.padding_idx).to(self.device)
            
            tgt_pos = torch.LongTensor(range(tgt.size(1))).to(self.device)
            tgt = self.embedding(tgt) + self.dec_pos_enc(tgt_pos)
                                               
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
            
            output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                        src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask).transpose(0,1)

            output = self.classifier(output)
            output = self.logsoftmax(output)

            return output

        else:
            memory = self.transformer.encoder(src.transpose(0,1), mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            if tgt is None: # Inferenece
                tgt = torch.LongTensor([[self.sos_id]]).to(self.device)
                tgt = tgt.repeat(src.size(0), 1)
                for di in range(self.max_length):
                    tgt_pos = torch.LongTensor(range(tgt.size(1))).to(self.device)
                    tgt_ = self.embedding(tgt) + self.dec_pos_enc(tgt_pos)
                    tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                    output = self.transformer.decoder(tgt_.transpose(0,1), memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask).transpose(0,1)
                    
                    output = self.classifier(output)
                    output = self.logsoftmax(output)
                    symbols = torch.max(output, -1)[1][:,-1].unsqueeze(1)
                    tgt = torch.cat((tgt,symbols),1)
            else: # Evaluate
                tgt_size = tgt.size(1)
                tgt = tgt[:,0].unsqueeze(1)
                for di in range(tgt_size):
                    tgt_pos = torch.LongTensor(range(tgt.size(1))).to(self.device)
                    tgt_ = self.embedding(tgt) + self.dec_pos_enc(tgt_pos)
                    tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                    output = self.transformer.decoder(tgt_.transpose(0,1), memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask).transpose(0,1)
                    
                    output = self.classifier(output)
                    output = self.logsoftmax(output)
                    symbols = torch.max(output, -1)[1][:,-1].unsqueeze(1)
                    tgt = torch.cat((tgt,symbols),1)
                
            return output
