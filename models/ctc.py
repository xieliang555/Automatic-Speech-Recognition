import torch
import torch.nn as nn
import torch.nn.functional as F


class CTC_ASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        nhid = config.model.ctc.nhid
        nlayer = config.model.ctc.nlayer
        
        ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Half-time dimension
        )
        self.rnn = nn.LSTM(1280, nhid, nlayer, 
            batch_first=True, bidirectional=True)
#         self.rnn_2 = nn.LSTM(
#             input_size=256, hidden_size=128, num_layers=1, 
#             batch_first=True, bidirectional=True)
        
        self.out = nn.Linear(256, config.data.vocabSize)
        
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

        self.rnn.flatten_parameters()
#         self.rnn_2.flatten_parameters()
#         if feature.shape[1] % 4 != 0:
#             feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        feature, _ = self.rnn(feature)
#         feature = feature[:,::2,:]
#         feature, _ = self.rnn_2(feature)
#         feature = feature[:,::2,:]
        logits = self.out(feature)
#         feat_len = feat_len//4
        return logits, feat_len
    
    
    