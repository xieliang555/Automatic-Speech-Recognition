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
import math
import yaml
import argparse
import numpy as np
from itertools import groupby

import sys
sys.path.append('..')
from utils import wer, AttrDict
from models.language_model import LM
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
    for batchIdx, batch in enumerate(trainLoader):
        inputs = batch['inputs'].cuda()
        targets = batch['targets'].cuda()

        optimizer.zero_grad()
        # noise weight
        if args.resume_training:
            hh_noise = torch.normal(0,config.model.ctc.sigma, size=net.encoder.rnn.weight_hh_l0.shape).cuda()
            ih_noise = torch.normal(0,config.model.ctc.sigma, size=net.encoder.rnn.weight_ih_l0.shape).cuda()
            net.encoder.rnn.weight_hh_l0.data.add_(hh_noise)
            net.encoder.rnn.weight_ih_l0.data.add_(ih_noise)
            
        outputs, _ = net(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        loss = criterion(outputs, targets.view(-1))
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        
        N = len(trainLoader) // 10
        if batchIdx % N == N-1:
            running_loss /= N
            print(f'epoch: {epoch} | batch: {batchIdx} | loss: {running_loss} | ppl: {math.exp(running_loss)}')
            running_loss = 0.0


def evaluate(net, devLoader, criterion):
    net.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            inputs = batch['inputs'].cuda()
            targets = batch['targets'].cuda()
            
            outputs, _ = net(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            loss = criterion(outputs, targets.view(-1))
            epoch_loss += loss.item()
            
    epoch_loss /= len(devLoader)      
    return epoch_loss, math.exp(epoch_loss)


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
    inputs: [N,L]
    targets: [N,L]
    '''
    txt_seqs, seqs_len = TEXT.process([item[1] for item in batch]) 
    inputs = txt_seqs[:,:-1]
    targets = txt_seqs[:,1:]
    return {'inputs':inputs, 'targets':targets}

trainLoader = DataLoader(
    trainSet, batch_size=config.training.BSZ,shuffle=True, pin_memory=True,
    collate_fn=my_collate, num_workers=0)
devLoader = DataLoader(
    devSet, batch_size=config.training.BSZ, shuffle=False, pin_memory=True, 
    collate_fn=my_collate, num_workers=0)


###############################################################################
# Define model
###############################################################################
net = LM(config).cuda()
if args.resume_training:
    ckpt = torch.load(config.data.trained_lm)
    start_epoch = ckpt['epoch']+1
    best_dev_ppl = ckpt['best_dev_ppl']
    net.load_state_dict(ckpt['net_state_dict'])
    print(f'resume training from epcoh {start_epoch} with best_dev_ppl {best_dev_ppl}')
else:
    start_epoch = 0
    best_dev_ppl = float('inf')
criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=config.data.pad_idx)
optimizer = optim.SGD(net.parameters(), lr=config.training.lr)

# summary(net, torch.zeros(2,500,26).cuda())
###############################################################################
# Training code
###############################################################################

for epoch in range(start_epoch, 1000):
    train(net, trainLoader, criterion, optimizer, epoch)
    epoch_loss, epoch_ppl = evaluate(net, devLoader, criterion)
    print(f'end of epoch {epoch}: dev loss {epoch_loss} | dev per {epoch_ppl}')
    
    if epoch_ppl < best_dev_ppl:
        best_dev_ppl = epoch_ppl
        torch.save({'net_state_dict':net.state_dict(), 
                    'epoch':epoch, 'best_dev_ppl':best_dev_ppl}, config.data.trained_lm)
        print('best model saved')
        
    







