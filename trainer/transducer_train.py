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
from warp_rnnt import rnnt_loss

import sys
sys.path.append('..')
from utils import wer, AttrDict
from models.transducer import Transducer_ASR
from datasets import TIMIT


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--device', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.device
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def train(net, trainLoader, optimizer, epoch):
    net.train()
    running_loss = 0.0
    for batchIdx, batch in enumerate(trainLoader):
        inputs = batch['feature'].cuda()
        inputs_len = batch['feat_len'].cuda()
        targets = batch['utterance'].cuda()
        targets_len = batch['utter_len'].cuda()
        
        optimizer.zero_grad()
        # noise weight
        if args.resume_training:
            hh_noise = torch.normal(0,config.model.ctc.sigma, size=net.encoder.rnn.weight_hh_l0.shape).cuda()
            ih_noise = torch.normal(0,config.model.ctc.sigma, size=net.encoder.rnn.weight_ih_l0.shape).cuda()
            net.encoder.rnn.weight_hh_l0.data.add_(hh_noise)
            net.encoder.rnn.weight_ih_l0.data.add_(ih_noise)
        
        loss = net(inputs, targets, inputs_len, targets_len)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        N = len(trainLoader) // 10
        if batchIdx % N == N-1:
            print(f'epoch: {epoch} | batch: {batchIdx} | loss: {running_loss/N}')
            running_loss = 0.0


def evaluate(net, devLoader):
    net.eval()
    epoch_per = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            inputs = batch['feature'].cuda()
            inputs_len = batch['feat_len'].cuda()
            targets = batch['utterance'].cuda()
            targets_len = batch['utter_len'].cuda()
            
            preds = net.best_path_decode(inputs, inputs_len)
            preds = [[k for k, _ in groupby(sent)] for sent in preds]
            targets = [u[u!=config.data.pad_idx] for u in targets]
            per = np.array([wer(*z) for z in zip(targets, preds)]).mean()
            epoch_per += per
            
    return epoch_per/len(devLoader)


###############################################################################
# Load data
###############################################################################
print('load dataset')
configfile = open('../config.yaml')
config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

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
    trainSet, batch_size=config.training.BSZ, shuffle=True, pin_memory=True,
    collate_fn=my_collate, num_workers=0)

devLoader = DataLoader(
    devSet, batch_size=config.training.BSZ, shuffle=False, pin_memory=True, 
    collate_fn=my_collate, num_workers=0)

###############################################################################
# Define model
###############################################################################
net = Transducer_ASR(config).cuda()
if args.resume_training:
    ckpt = torch.load(config.data.trained_transducer)
    start_epoch = ckpt['epoch']+1
    best_dev_per = ckpt['best_dev_per']
    net.load_state_dict(ckpt['net_state_dict'])
    print(f'resume training from epcoh {start_epoch} with best_dev_per {best_dev_per}')
else:
    start_epoch = 0
    best_dev_per = float('inf')
optimizer = optim.SGD(net.parameters(), lr=config.training.lr, momentum=config.training.momentum)

summary(net, 
        torch.zeros(1,10,123).cuda(), torch.zeros(1,15).long().cuda(), 
        torch.tensor([10]).long().cuda(), torch.tensor([15]).long().cuda())
###############################################################################
# Training code
###############################################################################

for epoch in range(start_epoch, 1000):
    train(net, trainLoader, optimizer, epoch)
    dev_per = evaluate(net, devLoader)
    print(f'end of epoch {epoch}: dev_per: {dev_per}')
    
    if dev_per < best_dev_per:
        best_dev_per = dev_per
        torch.save({'net_state_dict':net.state_dict(), 
                    'epoch':epoch, 'best_dev_per':best_dev_per}, 
                   config.data.trained_transducer)
        print('best model saved')
        
    







