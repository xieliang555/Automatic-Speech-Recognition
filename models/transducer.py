import torch
import torch.nn as nn
import torch.nn.functional as F

from warp_rnnt import rnnt_loss

from ctc import CTC_ASR
from language_model import LM


class Transducer_ASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        pretrained_encoder = config.data.pretrained_encoder
        pretrained_decoder = config.data.pretrained_decoder
        encoder_nout = config.model.transducer.encoder_nout
        decoder_nout = config.model.transducer.decoder_nout
        nhid = config.model.transducer.nhid
        vocabSize = config.data.vocabSize
        self.blank_idx = config.data.blank_idx
        
        self.encoder = CTC_ASR(config)
        if pretrained_encoder:
            ckpt = torch.load(pretrained_encoder)
            self.encoder.load_state_dict(ckpt['net_state_dict'])
        self.encoder.out = nn.Linear(self.encoder.out.in_features, encoder_nout)
        
        self.decoder = LM(config)
        if pretrained_decoder:
            ckpt = torch.load(pretrained_decoder)
            self.decoder.load_state_dict(ckpt['net_state_dict'])
        self.decoder.out = nn.Linear(self.decoder.out.in_features, decoder_nout)
        
        self.joint = nn.Linear(encoder_nout+decoder_nout, nhid)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(nhid, vocabSize)
    
    def forward(self, inputs, targets, inputs_len, targets_len):
        '''
        Args:
            inputs(acoustic feature): [N,T,D]
            targets(phoneme sequence): [N,T]
            inputs_len: [N]
            targets_len: [N]
        Return:
            outputs(predicted logits): [N,T,E]
        '''
        enc_state, input_len = self.encoder(inputs, inputs_len)
        targets = F.pad(targets, pad=[1,0], value=0)
        dec_state = self.decoder(targets)
        
        dec_state = dec_state.unsqueeze(1)
        enc_state = enc_state.unsqueeze(2)
        t = enc_state.size(1)
        u = dec_state.size(2)
        dec_state = dec_state.repeat([1,t,1,1])
        enc_state = enc_state.repeat([1,1,u,1])
        concat_state = torch.cat([enc_state, dec_state], dim=-1)
        
        # softmax?
        logits = self.out(self.tanh(self.joint(concat_state)))
        loss = rnnt_loss(logits, targets, inputs_len, targets_len, blank=int(blank_idx))
        return loss
    
    
    
    
    
    
    