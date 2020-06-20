import torch
import torch.nn as nn
import torch.nn.functional as F

from warp_rnnt import rnnt_loss
# from warprnnt_pytorch import RNNTLoss

from .ctc import CTC_ASR
from .language_model import LM



class Transducer_ASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        using_trained_ctc = config.data.using_trained_ctc
        using_trained_lm = config.data.using_trained_lm
        trained_ctc = config.data.trained_ctc
        trained_lm = config.data.trained_lm
        encoder_nout = config.model.transducer.encoder_nout
        decoder_nout = config.model.transducer.decoder_nout
        nhid = config.model.transducer.nhid
        self.vocabSize = config.data.vocabSize
        self.blank_idx = config.data.blank_idx
        
        self.encoder = CTC_ASR(config)
        if using_trained_ctc:
            ckpt = torch.load(trained_ctc)
            self.encoder.load_state_dict(ckpt['net_state_dict'])
        self.encoder.ctc_out = nn.Linear(self.encoder.ctc_out.in_features, encoder_nout)
        
        self.decoder = LM(config)
        if using_trained_lm:
            ckpt = torch.load(trained_lm)
            self.decoder.load_state_dict(ckpt['net_state_dict'])
        self.decoder.lm_out = nn.Linear(self.decoder.lm_out.in_features, decoder_nout)
        
        self.joint = nn.Linear(encoder_nout+decoder_nout, nhid)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(nhid, self.vocabSize)
#         self.crit = RNNTLoss()
    
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
        enc_state, inputs_len = self.encoder(inputs, inputs_len)
        dec_state = self.decoder(F.pad(targets, pad=[1,0,0,0], value=self.blank_idx))
        
        dec_state = dec_state.unsqueeze(1)
        enc_state = enc_state.unsqueeze(2)
        t = enc_state.size(1)
        u = dec_state.size(2)
        dec_state = dec_state.repeat([1,t,1,1])
        enc_state = enc_state.repeat([1,1,u,1])
        concat_state = torch.cat([enc_state, dec_state], dim=-1)
        
        # softmax?
        logits = self.out(self.tanh(self.joint(concat_state)))
        logits = F.log_softmax(logits, dim=-1)
        loss = rnnt_loss(logits, targets.int(), inputs_len.int(), targets_len.int(), blank=self.blank_idx)
#         loss = self.crit(logits, targets.int(), inputs_len.int(), targets_len.int())
        return loss.mean()
    
    
    
    
    
    
    