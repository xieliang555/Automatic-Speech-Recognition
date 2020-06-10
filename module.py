import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # bidirectional
        self.nout = 512*2
        
        ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Half-time dimension
        )
        self.rnn = nn.LSTM(
            input_size=1280, hidden_size=512, num_layers=5, 
            batch_first=True, bidirectional=True)
        
    def view_input(self, feature, feat_len):
        # downsample time because of max-pooling ops over time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        # fbank acoustic feature: 3*40
        feature = feature.view(bs, ts, 3, 40)
        feature = feature.transpose(1, 2)
        return feature, feat_len

    def forward(self, feature, feat_len):
        # Feature shape NxTxD -> N x C(num of delta) x T x D(acoustic feature dim: 40)
        feature, feat_len = self.view_input(feature, feat_len)
        feature = self.extractor(feature)
        # Nx128xT/4xD/4 -> NxT/4x128xD/4
        feature = feature.transpose(1, 2)
        #  N x T/4 x 128 x D/4 -> N x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], -1)
        feature, _ = self.rnn(feature)
        return feature, feat_len
    
    
    