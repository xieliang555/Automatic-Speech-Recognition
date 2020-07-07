import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from ctc import ResBlock


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        nhid = config.model.las.encoder.nhid
        nlayer = config.model.las.encoder.nlayer
        nout = config.model.las.decoder.nhid
        
        self.block1 = ResBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.block2 = ResBlock(64, 64)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.rnn = nn.GRU(640, nhid, nlayer, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(nhid*2, nout)
        
    def view_input(self, feature, feat_len):
        '''
        Args:
            feature: [N,T,3D]
            feat_len: [N]
        Return:
            feature: [N,3,T,D]
            feat_len: [N]
        '''
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
        '''
        Args:
            feature: [N,T,3D] 3D: 3(num of delta) * 41(fbank acoustic feature dim)
            feat_len: [N]
        Return:
            outputs: [N,T,H]
            hidden: [N,H] 
        '''
        feature, feat_len = self.view_input(feature, feat_len)
        feature = self.block1(feature)
        feature = self.pool1(feature)
        feature = self.block2(feature)
        feature = self.pool2(feature)
        # [N,64,T/4,D/4] -> [N,T/4,16D]
        feature = feature.transpose(1,2).view(feature.shape[0], feature.shape[1],-1)
        outputs, hidden = self.rnn(feature)
        # hidden[-1,:,:] and hidden[-2,:,:] means the top encoder 
        # forward and backward hidden states
        hidden = torch.tanh(self.fc(torch.cat(
            [hidden[-2,:,:],hidden[-1,:,:]],dim=-1)))
        return outputs, hidden
    
   
    
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_nhid = config.model.las.encoder.nhid
        dec_nhid = config.model.las.decoder.nhid
        atten_nhid = config.model.las.attention.nhid
        
        self.atten = nn.Linear(enc_ndim*2+dec_ndim, atten_nhid)
    
    def forward(self, decoder_hidden, encoder_outputs):
        '''
        Args:
            decoder_hidden: [N,H]
            encoder_outputs: [N,T,H]
        Return:
            attention: [N,T]
        '''
        t = encoder_outputs.size(1)
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1,t,1)
        energy = self.atten(torch.cat([repeated_decoder_hidden, encoder_outputs], dim=-1))
        energy = torch.tanh(energy)
        attention = F.softmax(torch.sum(energy, dim=-1), dim=-1)
        return attention
    
    
     
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocabSize = config.data.vocabSize
        nemd = config.model.las.decoder.nemd
        emd_drop = config.model.las.decoder.emd_drop
        dec_nhid = config.model.las.decoder.nhid
        enc_nhid = config.model.las.encoder.nhid
    
        self.attention = Attention(config)
        self.embedding = nn.Embedding(vocabSize, nemd)
        self.dropout = nn.Dropout(emd_drop)
        self.rnn = nn.GRU(nemd+2*enc_nhid, dec_nhid, batch_first=True)
        self.out = nn.Linear(nemd+2*enc_nhid+dec_nhid, vocabSize)
        
        
    def get_context_vector(decoder_hidden, encoder_outputs):
        '''
        Args:
            decoder_hidden: [N,H]
            encoder_outputs: [N,T,H]
        Return:
            context: [N,1,H]
        '''
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        context = torch.bmm(a, encoder_outputs)
        return context
   
    
    def forward(self, trg, decoder_hidden, encoder_outputs):
        '''
        Args:
            trg: [N,1]
            decoder_hidden: [N,H]
            encoder_outputs: [N,T,H]
        Return:
            output: [N,O]
            decoder_hidden: [N,H]
        '''
        embedded = self.dropout(self.embedding(trg))
        context = get_context_vector(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=-1)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        
        output = torch.cat([output, context, embedded], dim=2).suqeeze(1)
        output = self.out(output)
        return output, decoder_hidden.squeeze(0)
    
    
    
class LAS_ASR(nn.Moudel):
    def __init__(self, config):
        '''LAS: listener(Encoder), attention(Attention), speller(Decoder)'''
        super().__init__()
        vocabSize = config.data.vocabSize
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, feature, trg, teacher_forcing_ratio=0.5):
        '''
        Args:
            feature: [N,T,D]
            trg: [N,T]
        Return:
            outputs: [N,T,O]
        '''
        encoder_outputs, hidden = self.encoder(feature)
        N,T = trg.shape
        outputs = torch.zeros(T,N,vocabSize).cuda()
        
        output = torch.zeros([N,1],dtype=torch.long).cuda()
        for t in range(T):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = trg[:,t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs.transpose(0,1)
    
    
    
    
    
  