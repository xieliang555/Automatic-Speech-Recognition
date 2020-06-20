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
import argparse
import numpy as np
from itertools import groupby

import sys
sys.path.append('..')
from utils import wer, AttrDict
from models.ctc import CTC_ASR
from datasets import TIMIT


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--device', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.device
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def train(net, trainLoader, criterion, optimizer, epoch):
    net.train()
    running_loss = 0.0
    running_wer = 0.0
    for batchIdx, batch in enumerate(trainLoader):
        feature = batch['feature'].cuda()
        feat_len = batch['feat_len'].cuda()
        utterance = batch['utterance'].cuda()
        utter_len = batch['utter_len'].cuda()
        
        optimizer.zero_grad()
        logits, feat_len = net(feature, feat_len)
        log_logits = F.log_softmax(logits, dim=-1).transpose(0,1)
        # without downsampling loss-compute will be assuming
        loss = criterion(log_logits, utterance, feat_len, utter_len)
        running_loss += loss.item()
        preds = log_logits.max(-1)[1].transpose(0,1)
        preds = [[k for k, _ in groupby(s) if k!=config.data.blank_idx] for s in preds]
        utterance = [u[u!=config.data.pad_idx] for u in utterance]
        # long sequence lower computation
        running_wer += np.array([wer(*z) for z in zip(utterance, preds)]).mean()
        loss.backward()
        optimizer.step()
        
        N = len(trainLoader) // 10
        if batchIdx % N == N-1:
            print(f'epoch: {epoch} | batch: {batchIdx} | loss: {running_loss/N} | wer: {running_wer/N}')
            running_loss = 0.0
            running_wer = 0.0


def evaluate(net, devLoader, criterion):
    net.eval()
    epoch_loss = 0.0
    epoch_wer = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            feature = batch['feature'].cuda()
            feat_len = batch['feat_len'].cuda()
            utterance = batch['utterance'].cuda()
            utter_len = batch['utter_len'].cuda()
            
            logits, feat_len = net(feature, feat_len)
            log_logits = F.log_softmax(logits, dim=-1).transpose(0,1)
            loss = criterion(log_logits, utterance, feat_len, utter_len)
            epoch_loss += loss.item()
            preds = log_logits.max(-1)[1].transpose(0,1)
            preds = [[k for k, _ in groupby(s) if k!=config.data.blank_idx] for s in preds]
            utterance = [u[u!=config.data.pad_idx] for u in utterance]
            epoch_wer += np.array([wer(*z) for z in zip(utterance, preds)]).mean()
            
    return epoch_loss/len(devLoader), epoch_wer/len(devLoader)


###############################################################################
# Load data
###############################################################################
print('load data')
configfile = open('../config.yaml')
config=AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
trainSet = TIMIT(config.data.data_root, mode='train')
devSet = TIMIT(config.data.data_root, mode='test')

TEXT = Field(lower=True, include_lengths=True, 
             batch_first=True)
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
print(len(TEXT.vocab))
assert config.data.vocabSize == len(TEXT.vocab)
print(TEXT.vocab.stoi['<pad>'])
assert config.data.pad_idx == TEXT.vocab.stoi['<pad>']
print(TEXT.vocab.stoi['<blank>'])
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

trainLoader = DataLoader(
    trainSet, batch_size=config.training.BSZ,shuffle=True, pin_memory=True,
    collate_fn=my_collate, num_workers=0)
devLoader = DataLoader(
    devSet, batch_size=config.training.BSZ, shuffle=False, pin_memory=True, 
    collate_fn=my_collate, num_workers=0)

# for batchIdx, batch in enumerate(devLoader):
#     print(batch['feature'].shape)
#     print(batch['feat_len'])
#     print(batch['utterance'])
#     print(batch['utter_len'])
#     if batchIdx == 3:
#         break

###############################################################################
# Define model
###############################################################################
net = CTC_ASR(config).cuda()
if args.resume_training:
    ckpt = torch.load(config.data.trained_ctc)
    start_epoch = ckpt['epoch']+1
    best_dev_wer = ckpt['best_dev_wer']
    net.load_state_dict(ckpt['net_state_dict'])
    print(f'resume training from epcoh {start_epoch} with best_dev_wer {best_dev_wer}')
else:
    start_epoch = 0
    best_dev_wer = float('inf')
criterion = nn.CTCLoss(blank=config.data.blank_idx, zero_infinity=True)
optimizer = optim.SGD(net.parameters(), lr=config.training.lr, momentum=config.training.momentum)

# summary(net, torch.zeros(2,500,26).cuda())
###############################################################################
# Training code
###############################################################################

for epoch in range(start_epoch, 1000):
    train(net, trainLoader, criterion, optimizer, epoch)
    epoch_loss, epoch_wer = evaluate(net, devLoader, criterion)
    print(f'end of epoch {epoch}: dev loss {epoch_loss} | dev wer {epoch_wer}')
    
    if epoch_wer < best_dev_wer:
        best_dev_wer = epoch_wer
        torch.save({'net_state_dict':net.state_dict(), 
                    'epoch':epoch, 'best_dev_wer':best_dev_wer}, config.data.trained_ctc)
        print('best model saved')
        
    







