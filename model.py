import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Encoder



class ASR(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.encoder = Encoder(args)
        self.out = nn.Linear(self.encoder.nout, args.vocabSize+1)
        
    
    def forward(self, feature, feat_len):
        feature, feat_len = self.encoder(feature, feat_len)
        logits = F.log_softmax(self.out(feature), dim=-1).transpose(0,1)
        return logits, feat_len