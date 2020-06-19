import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field
from torchsummaryX import summary

import os
import argparse
import numpy as np
from itertools import groupby

from utils import wer
from models.ctc import CTC_ASR
from datasets import TIMIT


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root', type=str, default="/home/xieliang/Data",
    help='training and evaluating data root')
parser.add_argument('--BSZ', type=int, default=8, help='batch size')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument(
    '--num_workers', type=int, default=0, 
    help='number of process for loading data')
parser.add_argument('--use_cuda', default=False)
parser.add_argument('--vocabSize', type=int, default=40)
parser.add_argument('--pad_idx', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)



def evaluate(net, devLoader, criterion):
    net.eval()
    epoch_loss = 0.0
    epoch_wer = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            feature = batch['feature']
            feat_len = batch['feat_len']
            utterance = batch['utterance']
            utter_len = batch['utter_len']
            
            logits, feat_len = net(feature, feat_len)
            loss = criterion(logits, utterance, feat_len, utter_len)
            epoch_loss += loss.item()
            preds = logits.max(-1)[1].transpose(0,1)
            preds = [[k for k,_ in groupby(s) if k!=args.vocabSize] for s in preds]
            utterance = [u[u!=args.pad_idx] for u in utterance]
            print('utterance', utterance)
            print('preds', preds)
            epoch_wer += np.array([wer(*z) for z in zip(utterance, preds)]).mean()
            
        return epoch_loss/len(devLoader), epoch_wer/len(devLoader)


###############################################################################
# Load data
###############################################################################

trainSet = TIMIT(args.data_root, mode='train')
devSet = TIMIT(args.data_root, mode='test')

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
sents = ['iy', 'ix', 'eh', 'ae', 'ax', 'uw', 'uh',
         'ao', 'ey', 'ay', 'oy', 'aw', 'ow', 'er',
         'l', 'r', 'w', 'y', 'm', 'n', 'ng', 'v',
         'f', 'dh', 'th', 'z', 's', 'zh', 'jh', 'ch',
         'b', 'p', 'd', 'dx', 't', 'g', 'k', 'hh', 'h#']
sents = [[i] for i in sents]
TEXT.build_vocab(sents)
assert args.vocabSize == len(TEXT.vocab)
assert args.pad_idx == TEXT.vocab.stoi['<pad>']


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
    trainSet, batch_size=1,shuffle=True, pin_memory=args.use_cuda,
    collate_fn=my_collate, num_workers=args.num_workers)
devLoader = DataLoader(
    devSet, batch_size=1, shuffle=False, pin_memory=args.use_cuda, 
    collate_fn=my_collate, num_workers=args.num_workers)

###############################################################################
# Define model
###############################################################################
net = CTC_ASR(args)
ckpt = torch.load("/home/xieliang/Data/ctc1.pth", map_location=torch.device('cpu'))
print('epoch', ckpt['epoch'])
print('best_dev_wer', ckpt['best_dev_wer'])
net.load_state_dict(ckpt['net_state_dict'])
criterion = nn.CTCLoss(blank=args.vocabSize, zero_infinity=True)

epoch_loss, epoch_wer = evaluate(net, devLoader, criterion)
print(epoch_loss, epoch_wer)
        
    







