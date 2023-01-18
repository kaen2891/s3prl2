# -*- coding: cp949 -*-

import time
import os
import numpy as np
import argparse

#from apex.parallel import DistributedDataParallel as DDP
import builtins
import warnings
import Levenshtein as Lev

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

import random
import label_loader
from e2e_loader import *

from model import Model, DownstreamConfig
from transformers import AdamW
import math
#train.cumulative_batch_count = 0

summary = None

from functools import lru_cache
############
# CONSTANT #
############
EOS_token = None
index2char = None
noisy_token = None
noisy2_token = None


def get_pretrain_args():

    parser = argparse.ArgumentParser(description='Downstream hyperparameters')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--warmup', type=int, default=4000)
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2891)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--mask', type=float, default=0.2)
    parser.add_argument('--position_encoding_size', type=int, default=256)    
    
    # Audio parameters
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling Rate')
    parser.add_argument('--window_size', default=.025, type=float, help='Window size for spectrogram')
    parser.add_argument('--window_stride', default=.010, type=float, help='Window stride for spectrogram')    
    parser.add_argument('--mel_bin', default=80, type=int, help='Number of mel spectrogram frequency')
    parser.add_argument('--norm', type=int, default='0')    
    parser.add_argument('--hop_num', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=32)
    parser.add_argument('--sub_mode', type=str, default='max')
        
    
    # Downstream model hyperparameters 
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--d_hidden_size', type=int, default=256)    
    parser.add_argument('--d_enc_num_layers', type=int, default=6)
    parser.add_argument('--d_dec_num_layers', type=int, default=6)
    parser.add_argument('--d_num_attention_heads', type=int, default=8)
    parser.add_argument('--d_enc_intermediate_size', type=int, default=1024)
    parser.add_argument('--d_dec_intermediate_size', type=int, default=1024)
    parser.add_argument('--d_transformer_dropout_prob', type=float, default=0.1)
    
    parser.add_argument('--d_rnn_num_layers', type=int, default=3)
    parser.add_argument('--d_conv_dropout_prob', type=float, default=0.3)
    parser.add_argument('--max_spec_length', type=int, default=2500)
    parser.add_argument('--max_text_length', type=int, default=330)    
    parser.add_argument('--position_encoding', type=bool, default=False)
    parser.add_argument('--bidirectional', type=bool, default=False)    
    parser.add_argument('--subsampling', type=bool, default=False)
        
    # ML system parameters
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--n_gpus', default=-1, type=int, help='number of gpus for distributed training')
    parser.add_argument('--two_machine', default=0, type=int, help='special check')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
        
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to downstream checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--feature_path', default='None', type=str, metavar='PATH', help='path to downstream checkpoint (default: none)')
    parser.add_argument('--tensorboard_log', default='None', type=str, metavar='PATH', help='path to downstream checkpoint (default: none)')
    
    
    args = parser.parse_args()
    
    new_args = dict(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, warmup=args.warmup, ckpt=args.ckpt, 
        seed=args.seed, optim=args.optim, position_encoding_size=args.position_encoding_size)    
    
    audio_config = dict(sample_rate=args.sample_rate, frame_length=int(args.window_size*args.sample_rate), 
        frame_step=int(args.window_stride*args.sample_rate), mel_bin=args.mel_bin, preemphasis=0.097, 
        n_fft=nfft(int(args.window_size*args.sample_rate)), norm=args.norm, hop_num=args.hop_num, slice_num=args.slice_num, sub_mode=args.sub_mode)
    
    downstream_config = dict(device=args.device, hidden_size=args.d_hidden_size, num_enc_layers=args.d_enc_num_layers, num_dec_layers=args.d_dec_num_layers,
        num_attention_heads=args.d_num_attention_heads, enc_intermediate_size=args.d_enc_intermediate_size, dec_intermediate_size=args.d_dec_intermediate_size,
        transformer_dropout_prob=args.d_transformer_dropout_prob, rnn_num_layers=args.d_rnn_num_layers, conv_dropout_prob=args.d_conv_dropout_prob,
        max_spec_length=args.max_spec_length, max_text_length=args.max_text_length, position_encoding=args.position_encoding, bidirectional=args.bidirectional, subsampling=args.subsampling)
    
    sys_config = dict(workers=args.workers, world_size=args.world_size, rank=args.rank, n_gpus=args.n_gpus, two_machine=args.two_machine, dist_url=args.dist_url, dist_backend=args.dist_backend,
        downstream_resume=args.resume, start_epoch=args.start_epoch, print_freq=args.print_freq, feature_path=args.feature_path, tensorboard_log=args.tensorboard_log)
    
    return new_args, audio_config, downstream_config, sys_config


target_dict = dict()
target_dict_val = dict()

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

def load_targets_val(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict_val[key] = target



def main():

    #os.environ['NCCL_DEBUG'] = 'INFO'
    #os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    '''only multi'''
    #os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
    #os.environ['NCCL_IB_DISABLE'] = '1'
    args, audio_config, downstream_config, sys_config = get_pretrain_args()
    
    
    if audio_config['hop_num'] == 4:
        downstream_config['max_spec_length'] = int(downstream_config['max_spec_length'] / 4) #2000 / 4
    elif audio_config['hop_num'] == 8:
        downstream_config['max_spec_length'] = int(downstream_config['max_spec_length'] / 8) #2000 / 8
    elif audio_config['hop_num'] == 16:
        downstream_config['max_spec_length'] = int(downstream_config['max_spec_length'] / 16) # 2000 / 16
    
    
        
    # Fix seed and make backends deterministic    
    if args['seed'] is not None:
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training.'
            'This will turn on the CUDNN deterministic setting,'
            'which can slow down your training considerably!'
            'You may see unexpected behavior when restarting'
            'from checkpoints.')
    if sys_config['n_gpus'] == -1:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = sys_config['n_gpus']
    #print('ngpus_per_node', ngpus_per_node)
    sys_config['world_size'] = ngpus_per_node * sys_config['world_size']
    #print('sys_config[world_size]', sys_config['world_size'])
    
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, audio_config, downstream_config, sys_config))

class LabelSmoothingLoss(nn.Module):
    """
    ref: https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
    
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')
    
def nfft(frame_length):
    """ Number of FFT """
    return 2 ** (frame_length - 1).bit_length()

def get_model_state(model):
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def main_worker(gpu, ngpus_per_node, args, audio_config, downstream_config, sys_config):    
    if sys_config['two_machine'] != 0:
        gpu += 4
    print(gpu, type(gpu))
    print("Use GPU: {} for training".format(gpu))
    downstream_config['device'] = gpu
        
    text_label = '../train-1000hours_label.label'
    
    global index2char, SOS_token, EOS_token, noisy_token, noisy2_token, MASK_token, PAD_token
    
    scene_length = use_scene_length(audio_config['slice_num'])

    char2index, index2char = label_loader.load_label(text_label)
    SOS_token = char2index['<s>']    
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']    
    noisy_token = char2index['쟙']
    noisy2_token = char2index['쟢']    
    char2index['[MASK]'] = len(char2index)
    index2char[len(index2char)] = '[MASK]'
    MASK_token = char2index['[MASK]']    
        
    sys_config['rank'] = sys_config['rank'] * ngpus_per_node + downstream_config['device']
    #print('sys_config rank', sys_config['rank'])
    dist.init_process_group(backend=sys_config['dist_backend'], init_method=sys_config['dist_url'],
                            world_size=sys_config['world_size'], rank=sys_config['rank'])    
    torch.cuda.set_device(downstream_config['device'])
    
    
    if downstream_config['device'] != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    
        
    print('Initializing model...')
    #exit()
    
    downstream_model_config = DownstreamConfig(downstream_config)
        
    ft_model = Model(len(char2index), SOS_token, EOS_token, d_model=downstream_model_config.d_model, nhead=downstream_model_config.num_attention_heads, max_seq_len=downstream_model_config.max_spec_length, 
                                                         num_encoder_layers=downstream_model_config.num_enc_layers, num_decoder_layers=downstream_model_config.num_dec_layers,
                                                         enc_feedforward=downstream_model_config.enc_intermediate_size, dec_feedforward=downstream_model_config.dec_intermediate_size,
                                                         dropout=downstream_model_config.transformer_dropout_prob, max_length=downstream_model_config.max_text_length, padding_idx=PAD_token, mask_idx=MASK_token, 
                                                         device=downstream_model_config.device, num_lstm=downstream_model_config.num_rnn_layers, bi=downstream_model_config.isBidirectional, 
                                                         sub=downstream_model_config.isSubsample, pos=downstream_model_config.isPos)
    
    ft_model.cuda(downstream_config['device'])
    
    
    args['batch_size'] = int(args['batch_size'] / ngpus_per_node)  # calculate local batch size for each GPU
    sys_config['workers'] = int((sys_config['workers'] + ngpus_per_node - 1) / ngpus_per_node)    
    ft_model = torch.nn.parallel.DistributedDataParallel(ft_model, device_ids=[downstream_config['device']],find_unused_parameters=True)
    
    if args['optim'] == 'adamw':
        optimizer = torch.optim.AdamW(ft_model.parameters(), args['lr'], eps = 1e-08)
        
    if args['optim'] == 'adam':
        optimizer = Adam(ft_model.parameters(), args['lr'], eps = 1e-08)
    print('Optimizer ', optimizer)
    
    # optionally resume from a downstream checkpoint
    if sys_config['downstream_resume']:
        if os.path.isfile(sys_config['downstream_resume']):
            print("=> loading checkpoint '{}'".format(sys_config['downstream_resume']))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(downstream_config['device'])
            checkpoint = torch.load(sys_config['downstream_resume'], map_location=loc)
            sys_config['start_epoch'] = checkpoint['epoch']
            #pretrain_model.module.load_state_dict(checkpoint['TransformerModel'])
            ft_model.module.load_state_dict(checkpoint['Model'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(sys_config['downstream_resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(sys_config['downstream_resume']))
    
    cudnn.benchmark = True
    
    TRAIN_PATH = os.path.join(sys_config['feature_path'], 'KoSpeech_1000hour')    
    VALID_PATH = os.path.join(sys_config['feature_path'], 'valid_1000hour')
            
    train_data_list = os.path.join(TRAIN_PATH, 'data_list.csv')
    train_wav_paths = list()
    train_script_paths = list()
    
    with open(train_data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"
            train_wav_path, train_script_path = line.strip().split(',')
            train_wav_paths.append(os.path.join(TRAIN_PATH, train_wav_path))
            train_script_paths.append(os.path.join(TRAIN_PATH, train_script_path))
    print('number of train wav {}, number of train script {}'.format(len(train_wav_paths),len(train_script_paths)))
    train_target_path = os.path.join(TRAIN_PATH, 'train_label')
    load_targets(train_target_path)
    
    valid_data_list = os.path.join(VALID_PATH, 'data_list.csv')
    valid_wav_paths = list()
    valid_script_paths = list()
    
    with open(valid_data_list, 'r') as f: 
        for line in f:
            # line: "aaa.wav,aaa.label"
            valid_wav_path, valid_script_path = line.strip().split(',')
            valid_wav_paths.append(os.path.join(VALID_PATH, valid_wav_path))
            valid_script_paths.append(os.path.join(VALID_PATH, valid_script_path))
    print('number of valid wav {}, number of valid script {}'.format(len(valid_wav_paths),len(valid_script_paths)))
    valid_target_path = os.path.join(VALID_PATH, 'valid_label')
    load_targets_val(valid_target_path) ##
        
    train_dataset = SpectrogramDataset(config=audio_config, data_list=[train_wav_paths,train_script_paths],
                                   char2index=char2index, sos_id=SOS_token, eos_id=EOS_token, data_dir=TRAIN_PATH, target_dict=target_dict)
    
    valid_dataset = SpectrogramDataset(config=audio_config, data_list=[valid_wav_paths,valid_script_paths],
                                   char2index=char2index, sos_id=SOS_token, eos_id=EOS_token, data_dir=VALID_PATH, target_dict=target_dict_val)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)        
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    
    train_loader = AudioDataLoader(train_dataset, batch_size=args['batch_size'], shuffle=(train_sampler is None), num_workers=sys_config['workers'], pin_memory=True, sampler=train_sampler)
    valid_loader = AudioDataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=sys_config['workers'], pin_memory=True, sampler=valid_sampler)
    train_batch_num = len(train_loader)
    print('Train Batch Num ', train_batch_num)
    valid_batch_num = len(valid_loader)
    print('Valid Batch Num ', valid_batch_num)
    
    save_dir = './models/ckpt={}'.format(args['ckpt'])
        
    if sys_config['rank'] % ngpus_per_node == 0:
        from tensorboardX import SummaryWriter
        global summary
        if sys_config['tensorboard_log'] is not None:
            summary = SummaryWriter(sys_config['tensorboard_log'])
            print('summary loaded from {}'.format(sys_config['tensorboard_log']))
        else:
            summary = SummaryWriter()
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)       
    
    criterion = LabelSmoothingLoss(0.1, len(char2index), ignore_index=PAD_token).cuda(downstream_config['device'])
    
    train_begin = time.time()
    for epoch in range(sys_config['start_epoch'], args['epochs']):
        save_epoch = epoch+1
        save_model_name = os.path.join(save_dir, 'finetune_extract_epoch={}.pth.tar'.format(save_epoch))
        
        
        train_sampler.set_epoch(save_epoch)
        
        print('Epoch {} Training Starts'.format(save_epoch))        
        train_loss, train_cer = train(save_epoch, ft_model, train_batch_num, train_loader, criterion, optimizer, args, audio_config, downstream_config, sys_config, train_begin, ngpus_per_node)        
        print('Epoch %d (Training) Loss %0.4f CER %0.2f' % (save_epoch, train_loss, train_cer))        
        synchronize()
        
        print('Epoch {} Validation Starts'.format(save_epoch))
        valid_loss, valid_cer = evaluate(ft_model, valid_loader, criterion, args, audio_config, downstream_config, sys_config)        
        print('Epoch %d (Validation) Loss %0.4f CER %0.2f' % (save_epoch, valid_loss, valid_cer))                        
        synchronize()
        
        if sys_config['rank'] % ngpus_per_node == 0:
            state = {
                'epoch': save_epoch,
                'Model': ft_model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, save_model_name)
            print('model saved finished at {}'.format(save_model_name))
                        
            summary.add_scalar('train_loss', train_loss, save_epoch)
            summary.add_scalar('train_cer', train_cer, save_epoch)
            summary.add_scalar('valid_loss', valid_loss, save_epoch)
            summary.add_scalar('valid_cer', valid_cer, save_epoch)
            
            with open(os.path.join(save_dir, 'valid_cer_results.txt'), 'a') as f:
                for_write = 'epoch = {}, valid_cer = {}\n'.format(save_epoch, valid_cer)
                f.write(for_write)
        
        print('Shuffling batches...')
    
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()    

def decay_learning_rate(init_lr, it, iter_per_epoch, start_epoch, warmup):
    warmup_threshold = warmup
    step = start_epoch * iter_per_epoch + it + 1
    decayed_lr = init_lr * warmup_threshold ** 0.5 * min(step * warmup_threshold**-1.5, step**-0.5)
    return decayed_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(epoch, ft_model, total_batch_size, data_loader, criterion, optimizer, args, audio_config, downstream_config, sys_config, train_begin, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    cers = AverageMeter('CER', ':.3e')
        
    progress = ProgressMeter(total_batch_size, [batch_time, data_time, losses, cers], prefix="Epoch: [{}]".format(epoch))
        
    ft_model.train()

    print('train() start')
    
    end = time.time()
    begin = epoch_begin = time.time()

    for i, (data) in enumerate(data_loader):

        data_time.update(time.time() - end)

        for param_group in optimizer.param_groups:
            param_group['lr'] = decay_learning_rate(args['lr'], i, total_batch_size, epoch-1, args['warmup'])

        feats, scripts = data ####### from dataloader
        if feats.dim() == 3:
            feats = feats.unsqueeze(0)
            scripts = feats.unsqueeze(0)
                
        feats = feats.cuda(downstream_config['device'])        
        scripts = scripts.cuda(downstream_config['device'])
        target = scripts[:, 1:].clone()
        
        if epoch <= 3: 
            mask = 0.05
        elif epoch <= 6:
            mask = 0.10
        elif epoch <= 9:
            mask = 0.15
        else:
            mask = 0.20
                
        logit = ft_model(feats, scripts[:,:-1], audio_config['hop_num'], audio_config['slice_num'], audio_config['sub_mode'], feats.size(-1), mode='train', mask=mask)        
                
        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        
        
                
        dist, length, _, _, ref, hyp = get_distance(target, y_hat)        
        cer_batch = dist / length
        
        cers.update(cer_batch, feats.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 5)
        
        if math.isnan(grad_norm):
            print('Error : grad norm is NaN')
        else:
            optimizer.step()
            losses.update(loss.item(), feats.size(0))
            #total_loss += loss.item()
            #total_num += feats.size(0)
        
        #optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % sys_config['print_freq'] == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0
            progress.display(i)
            print('elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(elapsed, epoch_elapsed, train_elapsed))
            print('ref', ref)
            print('hyp', hyp)
            
            begin = time.time()
        
        total_steps = ((epoch-1) * total_batch_size) + i + 1
        if sys_config['rank'] % ngpus_per_node == 0:
            global summary
            
            summary.add_scalar('train_loss_steps', losses.avg, total_steps)
            summary.add_scalar('train_cer_steps', cers.avg, total_steps)                   
            summary.add_scalar('learning_rate', param_group['lr'], total_steps)

    print('train() completed')
    return losses.avg, cers.avg


def evaluate(ft_model, data_loader, criterion, args, audio_config, downstream_config, sys_config):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    cers = AverageMeter('CER', ':.3e')
        
    progress = ProgressMeter(len(data_loader), [batch_time, losses, cers], prefix="Evaluation: ")
        
    ft_model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(data_loader):
            
            feats, scripts = data ####### from dataloader
            if feats.dim() == 3:
                feats = feats.unsqueeze(0)
                scripts = feats.unsqueeze(0)
            feats = feats.cuda(downstream_config['device'])        
            scripts = scripts.cuda(downstream_config['device'])
            target = scripts[:, 1:].clone()
                                                
            logit = ft_model(feats, scripts[:,:-1], audio_config['hop_num'], audio_config['slice_num'], audio_config['sub_mode'], feats.size(-1))
            y_hat = logit.max(-1)[1]            
    
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            
            losses.update(loss.item(), feats.size(0))
                    
            dist, length, _, _, ref, hyp = get_distance(target, y_hat)        
            cer_batch = dist / length
            
            cers.update(cer_batch, feats.size(0))
    
    print('Loss {loss.avg:.3f} CER {cer.avg:.3f}'.format(loss=losses, cer=cers))
    return losses.avg, cers.avg



def label_to_string(labels):
    global index2char, EOS_token, noisy_token, noisy2_token
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == 1618: #l
                continue
            elif i.item() == 1620: #b
                continue
            
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == noisy_token: #l
                    continue
                elif j.item() == noisy2_token: #b
                    continue
                
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def label_to_string_origin(labels):
    global EOS_token, index2char
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents


def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 


def get_distance(ref_labels, hyp_labels):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i]) #string
        hyp = label_to_string(hyp_labels[i]) #predict
        ref_origin = label_to_string_origin(ref_labels[i])
        hyp_origin = label_to_string_origin(hyp_labels[i])
        
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 
        '''
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
        '''
    return total_dist, total_length, ref, hyp, ref_origin, hyp_origin



if __name__ == '__main__':
    main()
