import torch
import torch.nn as nn
import torch.nn.functional as F


class Listener(nn.Module):
    def __init__(self, args):
        super().__init__()
        nhid = args.encoder_nhid
        nlayer = args.encoder_nlayer
        nout = args.decoder_nhid
        
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) 
        )
        self.rnn = nn.LSTM(
            input_size=1280, hidden_size=nhid, 
            num_layers=nlayer, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(nhid*2, nout)
    
    def forward(self, feature):
        # fbank acoustic feature dim 3(num of delta) * 40
        # [N,T,3D] -> [N,T,3,D] -> [N,3,T,D]
        n,t,d = feature.shape
        feature = feature.view(n,t,3,40).transpose(1,2)
        feature = self.extractor(feature)
        # [N,128,T/4,D/4] -> [N,T/4,128,D/4] -> [N,T/4,32D]
        feature = feature.transpose(1,2).view(feature.shape[0], feature.shape[1],-1)
        outputs, hidden = self.rnn(feature)
        # hidden[-1,:,:] and hidden[-2,:,:] means the top encoder 
        # forward and backward hidden states
        hidden = torch.tanh(self.fc(torch.cat(
            [hidden[-2,:,:],hidden[-1,:,:]],dim=-1))).unsqueeze(0)
        return outputs, hidden
    
   
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self):
        pass
    
    
     
class Speller(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocabSize = args.vocabSize
        nemd = args.decoder_nemd
        drop_ratio = args.drop_ratio
        
        self.embedding = nn.Embedding(vocabSize, nemd)
        self.dropout = nn.Dropout(drop_ratio)
        # decoder using unstacked rnn (one layer)
        
        
    def get_context_vector():
        pass
   
    
    def forward(self, trg, decoder_hidden, encoder_outputs):
        '''
        trg: [N]
        '''
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        context = get_context_vector()
        
        pass
    
    
    
class LAS_ASR(nn.Moudel):
    def __init__(self):
        super().__init__()
        self.encoder = Listener()
        self.decoder = Speller()
    
    def forward(self, feature, trg):
        '''
        feature: [N, T, D]
        trg: [N, T]
        '''
        encoder_outputs, hidden = self.encoder(feature)
        
        
        pass
    
    
    
    
    
    