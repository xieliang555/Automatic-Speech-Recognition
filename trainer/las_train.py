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
from models.las import LAS_ASR
from datasets import TIMIT


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--device', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.device
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def train(net, trainLoader, criterion, optimizer):
    net.train()
    running_loss = 0.0
    running_per = 0.0
    for batchIdx, batch in enumerate(trainLoader):
        feature = batch['feature'].cuda()
        feat_len = batch['feat_len'].cuda()
        utterance = batch['utterance'].cuda()
        utter_len = batch['utter_len'].cuda()
        
        optimizer.zero_grad()
        logits = net(feature, utterance)
        loss = criterion(logits.view(-1, log_logits.size(-1)), utterance.view(-1))
        running_loss += loss.item()
        preds = logits.max(-1)[1]
        utterance = [u[u!=config.data.pad_idx] for u in utterance]
        # long sequence lower computation
        running_per += np.array([wer(*z) for z in zip(utterance, preds)]).mean()
        loss.backward()
        optimizer.step()
        
        N = len(trainLoader) // 10
        if batchIdx % N == N-1:
            print(f'batch: {batchIdx} | loss: {running_loss/N} | per: {running_per/N}')
            running_loss = 0.0
            running_per = 0.0


def evaluate(net, devLoader, criterion):
    net.eval()
    epoch_loss = 0.0
    epoch_per = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            feature = batch['feature'].cuda()
            feat_len = batch['feat_len'].cuda()
            utterance = batch['utterance'].cuda()
            utter_len = batch['utter_len'].cuda()
            
            logits = net(feature, utterance, teacher_forcing_ratio=0)
            loss = criterion(logits.view(-1, logits.size(-1)), utterance.view(-1))
            epoch_loss += loss.item()
            preds = logits.max(-1)[1]
            utterance = [u[u!=config.data.pad_idx] for u in utterance]
            epoch_per += np.array([wer(*z) for z in zip(utterance, preds)]).mean()
            
    return epoch_loss/len(devLoader), epoch_per/len(devLoader)


###############################################################################
# Load data
###############################################################################
print('load data')
configfile = open('../config.yaml')
config=AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
trainSet = TIMIT(config.data.data_root, mode='train')
devSet = TIMIT(config.data.data_root, mode='test')

TEXT = Field(lower=True, include_lengths=True, batch_first=True, unk_token=None)

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

trainLoader = DataLoader(
    trainSet, batch_size=config.training.BSZ,shuffle=True, pin_memory=True,
    collate_fn=my_collate, num_workers=0)
devLoader = DataLoader(
    devSet, batch_size=config.training.BSZ, shuffle=False, pin_memory=True, 
    collate_fn=my_collate, num_workers=0)


###############################################################################
# Define model
###############################################################################
net = LAS_ASR(config).cuda()
if args.resume_training:
    ckpt = torch.load(config.data.trained_las)
    start_epoch = ckpt['epoch']+1
    best_dev_per = ckpt['best_dev_per']
    net.load_state_dict(ckpt['net_state_dict'])
    print(f'resume training from epcoh {start_epoch} with best_dev_per {best_dev_per}')
else:
    start_epoch = 0
    best_dev_per = float('inf')
criterion = nn.CrossEntropyLoss(ignore_index=config.data.pad_idx)
optimizer = optim.SGD(net.parameters(), lr=config.training.lr, momentum=config.training.momentum)

summary(net, torch.zeros(2,100,123).cuda(), torch.zeros(2,10))


###############################################################################
# Training code
###############################################################################
for epoch in range(start_epoch, 1000):
    train(net, trainLoader, criterion, optimizer)
    epoch_loss, epoch_per = evaluate(net, devLoader, criterion)
    print(f'end of epoch {epoch}: dev loss {epoch_loss} | dev per {epoch_per}')
    
    if epoch_per < best_dev_per:
        best_dev_per = epoch_per
        torch.save({'net_state_dict':net.state_dict(), 
                    'epoch':epoch, 'best_dev_per':best_dev_per}, config.data.trained_las)
        print('best model saved')
        
    







