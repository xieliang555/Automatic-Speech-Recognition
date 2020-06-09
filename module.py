import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        if args.extractor == 'vgg':
            pass
        elif args.extractor == 'cnn1d':
            pass
    
    
    def forward(self):
        pass
    
    
    