import torch
import torch.nn as nn
import torch.nn.functional as F



class ResBlock(nn.Module):
    '''Easy implementation of BasicBlock in ResNet'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=1)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        if self.in_channels != self.out_channels:
            identity = self.downsample(identity)
            
        out = self.relu2(x + identity)
        return out
    


class CTC_ASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        nhid = config.model.ctc.nhid
        nlayer = config.model.ctc.nlayer
        vocabSize = config.data.vocabSize
        
        self.block1 = ResBlock(3,64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.block2 = ResBlock(64,64)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.rnn = nn.LSTM(640, nhid, nlayer, 
            batch_first=True, bidirectional=True)
        
        self.ctc_out = nn.Linear(nhid*2, vocabSize)
        
    def view_input(self, feature, feat_len):
        # downsample time because of max-pooling ops over time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        # fbank acoustic feature: 3*41
        feature = feature.view(bs, ts, 3, 41)
        feature = feature.transpose(1, 2)
        return feature, feat_len

    def forward(self, feature, feat_len):
        # Feature shape NxTxD -> N x C(num of delta) x T x D(acoustic feature dim: 41)
        feature, feat_len = self.view_input(feature, feat_len)
        feature = self.block1(feature)
        feature = self.pool1(feature)
        feature = self.block2(feature)
        feature = self.pool2(feature)
        # [N,64,T/4,D/4] -> [N,T/4,16D]
        feature = feature.transpose(1, 2)
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], -1)
        self.rnn.flatten_parameters()
        feature, _ = self.rnn(feature)
        
        logits = self.ctc_out(feature)
        return logits, feat_len
    
    
    