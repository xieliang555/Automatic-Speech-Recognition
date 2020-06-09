import torch
import torch.nn as nn

from module import Encoder



class ASR(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.encoder = Encoder(args)
        
    
    def forward(self):
        pass