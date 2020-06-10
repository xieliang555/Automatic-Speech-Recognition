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

from model import ASR


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root', type=str, default="/home/xieliang/Data",
    help='training and evaluating data root')
parser.add_argument('--BSZ', type=int, default=8, help='batch size')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument(
    '--num_workers', type=int, default=0, 
    help='number of process for loading data')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--vocabSize', type=int, default=2467)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']="2"
torch.manual_seed(args.seed)
if args.use_cuda:
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
        loss = criterion(logits, utterance, feat_len, utter_len)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        N = len(trainLoader) // 10
        if batchIdx % N == N-1:
            print(f'epoch: {epoch} | batch: {batchIdx} | loss: {running_loss/N}')
            running_loss = 0.0


def evaluate(net, devLoader, criterion):
    net.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            feature = batch['feature'].cuda()
            feat_len = batch['feat_len'].cuda()
            utterance = batch['utterance'].cuda()
            utter_len = batch['utter_len'].cuda()
            
            logits, feat_len = net(feature, feat_len)
            loss = criterion(logits, utterance, feat_len, utter_len)
            epoch_loss += loss.item()
            
        return epoch_loss/len(devLoader)


###############################################################################
# Load data
###############################################################################
trainSet = torchaudio.datasets.LIBRISPEECH(
    args.data_root, url='train-clean-100',
    folder_in_archive='LibriSpeech', download=True)
devSet = torchaudio.datasets.LIBRISPEECH(
    args.data_root, url='dev-clean',
    folder_in_archive='LibriSpeech', download=True)

TEXT = Field(lower=True, include_lengths=True, batch_first=True)
sents = [s[2].lower().split() for s in trainSet]
TEXT.build_vocab(sents, min_freq=1)
assert args.vocabSize == len(TEXT.vocab)

# using gpu to calculate acoustic feature ?
def my_collate(batch):
    '''
    feature: [N,T,120]
    feat_len: [N]
    utterance: [N,L]
    utter_len: [N]
    '''
    waveform = [item[0] for item in batch]  
    sample_rate = [item[1] for item in batch] 
    feature = []  
    feat_len = [] 
    for w in waveform:
        # w: [1,116400]
        # get Fbank feature
        f = torchaudio.compliance.kaldi.fbank(
            w, sample_frequency=sample_rate[0], num_mel_bins=40)
        # add deltas
        d1 = torchaudio.functional.compute_deltas(f)
        d2 = torchaudio.functional.compute_deltas(d1)
        f = torch.cat([f, d1, d2], dim=-1)
        # CMVN normalization
        mean = f.mean(0, keepdim=True)
        std = f.std(0, keepdim=True)
        f = (f-mean) / (std+1e-10)
        
        feature.append(f)
        feat_len.append(len(f))
    feature = pad_sequence(feature, batch_first=True)
    feat_len = torch.tensor(feat_len)
    
    utterance = [item[2].lower().split() for item in batch]      
    utterance, utter_len = TEXT.process(utterance)                 
    
    return {'feature':feature, 'feat_len': feat_len, 
            'utterance':utterance, 'utter_len': utter_len}

trainLoader = DataLoader(
    trainSet, batch_size=args.BSZ,shuffle=True, pin_memory=args.use_cuda,
    collate_fn=my_collate, num_workers=args.num_workers)
devLoader = DataLoader(
    devSet, batch_size=args.BSZ, shuffle=False, pin_memory=args.use_cuda, 
    collate_fn=my_collate, num_workers=args.num_workers)


###############################################################################
# Define model
###############################################################################
net = ASR(args).cuda()
criterion = nn.CTCLoss(blank=args.vocabSize)
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)


###############################################################################
# Training code
###############################################################################
for epoch in range(1000):
    # ?
    train(net, devLoader, criterion, optimizer, epoch)
    epoch_loss = evaluate(net, devLoader, criterion)
    print(f'end of epoch {epoch}: dev loss {epoch_loss}')
    







