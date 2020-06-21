import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field
from torchsummaryX import summary

import os
import yaml
import numpy as np
from itertools import groupby
from warp_rnnt import rnnt_loss

import sys
sys.path.append('..')
from utils import wer, AttrDict
from models.transducer import Transducer_ASR
from datasets import TIMIT




def evaluate(net, devLoader):
    net.eval()
    epoch_per = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            inputs = batch['feature']
            inputs_len = batch['feat_len']
            targets = batch['utterance']
            targets_len = batch['utter_len']
            
            preds = net.best_path_decode(inputs, inputs_len)
            preds = [[k for k, _ in groupby(sent)] for sent in preds]
            targets = [u[u!=config.data.pad_idx] for u in targets]
            per = np.array([wer(*z) for z in zip(targets, preds)]).mean()
            epoch_per += per
            print(f'evaluate: batchIdx {batchIdx} | per {per}')
            
    return epoch_per/len(devLoader)


###############################################################################
# Load data
###############################################################################
print('load dataset')
configfile = open('../config.yaml')
config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
devSet = TIMIT(config.data.data_root, mode='test')


TEXT = Field(lower=True, include_lengths=True, batch_first=True, unk_token=None)
# sents = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 
#          'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 
#          'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 
#          'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 
#          'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 
#          'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 
#          'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 
#          'ux', 'v', 'w', 'y', 'z', 'zh']

# 61 target phone mapped to 39
# ref: https://github.com/zzw922cn/Automatic_Speech_Recognition
print('build vocab')
sents = ['iy', 'ix', 'eh', 'ae', 'ax', 'uw', 'uh',
         'ao', 'ey', 'ay', 'oy', 'aw', 'ow', 'er',
         'l', 'r', 'w', 'y', 'm', 'n', 'ng', 'v',
         'f', 'dh', 'th', 'z', 's', 'zh', 'jh', 'ch',
         'b', 'p', 'd', 'dx', 't', 'g', 'k', 'hh', 'h#']
sents = [[i] for i in sents]
TEXT.build_vocab(sents, specials=['<blank>'])
assert config.data.vocabSize == len(TEXT.vocab)
assert config.data.pad_idx == TEXT.vocab.stoi['<pad>']
assert config.data.blank_idx == TEXT.vocab.stoi['<blank>']

def my_collate(batch):
    '''
    feature: [N,T,120]
    feat_len: [N]
    utterance: [N,L]
    utter_len: [N]
    '''
    feature = [item[0] for item in batch]  
    feat_len = torch.tensor([len(f) for f in feature])
    feature = pad_sequence(feature, batch_first=True)
    utterance = [item[1] for item in batch]      
    utterance, utter_len = TEXT.process(utterance)                 
    return {'feature':feature, 'feat_len': feat_len, 
            'utterance':utterance, 'utter_len': utter_len}


devLoader = DataLoader(
    devSet, batch_size=config.training.BSZ, shuffle=False, pin_memory=False, 
    collate_fn=my_collate, num_workers=0)


###############################################################################
# Load model
###############################################################################
net = Transducer_ASR(config)
ckpt = torch.load(config.data.trained_transducer, map_location=torch.device('cpu'))
start_epoch = ckpt['epoch']+1
best_dev_per = ckpt['best_dev_per']
net.load_state_dict(ckpt['net_state_dict'])
print(f'epoch: {start_epoch} | best_dev_per: {best_dev_per}')


###############################################################################
# evaluating
###############################################################################

dev_per = evaluate(net, devLoader)
print(f'test_dev_per:, {dev_per}')
        
    







