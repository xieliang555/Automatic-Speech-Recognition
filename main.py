import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root', type=str, default="/home/xieliang/Data",
    help='training and evaluating data root')
parser.add_argument('--BSZ', type=int, default=2, help='batch size')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument(
    '--extractor', type=str, default='vgg', 
    help='feature extractor type: [vgg, cnn1d]')
parser.add_argument(
    '--num_workers', type=int, default=0, 
    help='number of process for loading data')
parser.add_argument('--use_cuda', default=False)
args = parser.parse_args()

torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)


def train():
    pass


def evaluate():
    pass


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
for batchIdx, batch in enumerate(devLoader):
    print(batch['feature'].shape)
    print(batch['feat_len'])
    print(batch['utterance'])
    print(batch['utter_len'])
    
    if batchIdx == 0:
        break




###############################################################################
# Training code
###############################################################################








