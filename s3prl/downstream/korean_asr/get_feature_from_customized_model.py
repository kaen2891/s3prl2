# -*- coding: cp949 -*-

import time
import os
import numpy as np
import argparse
#from apex.parallel import DistributedDataParallel as DDP
import builtins
import warnings

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
from feature_extract_loader import *
import s3prl.hub as hub
from glob import glob


def get_pretrain_args():

    parser = argparse.ArgumentParser(description='Downstream hyperparameters')
    parser.add_argument('--input_dir', default='/NasData/junewoo/', type=str, metavar='PATH')
    parser.add_argument('--save_feature_dir', default='/NasData/junewoo/', type=str, metavar='PATH')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2891)
    parser.add_argument('--max_sequence_length', type=int, default=25)
            
    # ML system parameters
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--device', type=int, default=None)
        
    parser.add_argument('--pretrain_resume', default='None', type=str, metavar='PATH', help='path to upstream checkpoint (default: none)')
    parser.add_argument('--model_name', default='None', type=str, metavar='PATH', help='path to downstream checkpoint (default: none)')    
    
    
    args = parser.parse_args()
    
    new_args = dict(input_feature=args.input_dir, feature_save_dir=args.save_feature_dir, batch_size=args.batch_size, seed=args.seed, max_sequence_length=args.max_sequence_length)    
            
    sys_config = dict(workers=args.workers, world_size=args.world_size, rank=args.rank, dist_url=args.dist_url, dist_backend=args.dist_backend,
        pretrain_resume=args.pretrain_resume, pretrained_model_name=args.model_name, device=args.device)
    
    return new_args, sys_config


def main():

    #os.environ['NCCL_DEBUG'] = 'INFO'
    #os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    '''only multi'''
    #os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
    #os.environ['NCCL_IB_DISABLE'] = '1'
    args, sys_config = get_pretrain_args()
        
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
    
    ngpus_per_node = torch.cuda.device_count()
    sys_config['world_size'] = ngpus_per_node * sys_config['world_size']
    
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, sys_config))


def nfft(frame_length):
    """ Number of FFT """
    return 2 ** (frame_length - 1).bit_length()

def main_worker(gpu, ngpus_per_node, args, sys_config):    
    print("Use GPU: {} for training".format(gpu))
    sys_config['device'] = gpu
    
    upstream = getattr(hub, sys_config['pretrained_model_name'])(sys_config['pretrain_resume'])
    sys_config['rank'] = sys_config['rank'] * ngpus_per_node + sys_config['device']
    dist.init_process_group(backend=sys_config['dist_backend'], init_method=sys_config['dist_url'],
                            world_size=sys_config['world_size'], rank=sys_config['rank'])    
    torch.cuda.set_device(sys_config['device'])
        
    '''
    if sys_config['device'] != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    '''
        
    print('Initializing model...')
    
    upstream.cuda(sys_config['device'])
    
    args['batch_size'] = int(args['batch_size'] / ngpus_per_node)  # calculate local batch size for each GPU
    sys_config['workers'] = int((sys_config['workers'] + ngpus_per_node - 1) / ngpus_per_node)
    pretrained_model = torch.nn.parallel.DistributedDataParallel(upstream, device_ids=[sys_config['device']])
        
       
    TRAIN_PATH = os.path.join(args['input_feature'], 'KoSpeech_1000hour')    
    VALID_PATH = os.path.join(args['input_feature'], 'valid_1000hour')
    
    train_save_path = os.path.join(args['feature_save_dir'], 'KoSpeech_1000hour')
    valid_save_path = os.path.join(args['feature_save_dir'], 'valid_1000hour')
    
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    
    if not os.path.exists(valid_save_path):
        os.makedirs(valid_save_path)
    
    train_wavs = sorted(glob(os.path.join(TRAIN_PATH, '*.wav')))
    valid_wavs = sorted(glob(os.path.join(VALID_PATH, '*.wav')))
    
    print('number of train wav {}, number of valid wav {}'.format(len(train_wavs),len(valid_wavs)))
            
    train_dataset = SpectrogramDataset(wav_list=train_wavs)
    valid_dataset = SpectrogramDataset(wav_list=valid_wavs)    
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)        
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    
    train_loader = AudioDataLoader(train_dataset, batch_size=args['batch_size'], shuffle=(train_sampler is None), num_workers=sys_config['workers'], pin_memory=True, sampler=train_sampler)
    valid_loader = AudioDataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=sys_config['workers'], pin_memory=True, sampler=valid_sampler)
    
    train_batch_num = len(train_loader)
    print('Train Batch Num ', train_batch_num)
    valid_batch_num = len(valid_loader)
    print('Valid Batch Num ', valid_batch_num)
        
       
    train_sampler.set_epoch(1)        
    train(args, pretrained_model, train_save_path, train_loader, sys_config)
    print('training finished')
    synchronize()        
    
    evaluate(args, pretrained_model, valid_save_path, valid_loader, sys_config)
    print('eval finished')
    
    
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

def train(args, pretrained_model, save_dir, data_loader, sys_config):
    print('Training sets feature extraction starts')
    pretrained_model.eval()

    with torch.no_grad():
    
        for i, (data) in enumerate(data_loader):    
            feats, scripts = data ####### from dataloader
            
            file_save_name = os.path.join(save_dir, scripts[0])
            if os.path.isfile(file_save_name+'.npy'):
                print('{} is already exists'.format(file_save_name))
                continue
            else:
                feats = feats.unsqueeze(axis=0)
                
                if feats.size(-1) > int(16000*args['max_sequence_length']):
                    feats = feats[:, :int(16000*args['max_sequence_length'])]
                    print('resize, after feats', feats.size())
                
                feats = feats.cuda(sys_config['device'])
                
                with torch.no_grad():
                    features = pretrained_model(feats)
                    outputs = features['last_hidden_state']
                    print('outputs feats size', outputs.size())
                                    
                    for i in range(len(outputs)):
                        features = outputs[i].permute(1, 0)
                        new_features = features.cpu()
                        
                        np.save(os.path.join(save_dir, scripts[i]), new_features)
                                    

def evaluate(args, pretrained_model, save_dir, data_loader, sys_config):
    print('Validatation sets feature extraction starts')
    pretrained_model.eval()

    with torch.no_grad():
    
        for i, (data) in enumerate(data_loader):    
            feats, scripts = data ####### from dataloader
            
            file_save_name = os.path.join(save_dir, scripts[0])
            if os.path.isfile(file_save_name+'.npy'):
                print('{} is already exists'.format(file_save_name))
                continue
            else:
                feats = feats.unsqueeze(axis=0)
                
                if feats.size(-1) > int(16000*args['max_sequence_length']):
                    feats = feats[:, :int(16000*args['max_sequence_length'])]
                    print('resize, after feats', feats.size())
                
                feats = feats.cuda(sys_config['device'])
                
                with torch.no_grad():
                    features = pretrained_model(feats)
                    outputs = features['last_hidden_state']
                    print('outputs feats size', outputs.size())
                                    
                    for i in range(len(outputs)):
                        features = outputs[i].permute(1, 0)
                        new_features = features.cpu()
                        
                        np.save(os.path.join(save_dir, scripts[i]), new_features)



if __name__ == '__main__':
    main()
