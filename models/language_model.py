import torch
import torch.nn as nn
from torch.autograd import Variable


def embedded_dropout(embedding, inputs, dropout=0.1, scale=None):
    if dropout:
        mask = embedding.weight.data.new().resize_(embedding.weight.size(0), 1).bernoulli_(1-dropout)
        mask = mask.expand_as(embedding.weight)/(1-dropout)
        masked_embed_weight = mask * embedding.weight
    else:
        masked_embed_weight = embedding.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    padding_idx = embedding.padding_idx
    if padding_idx is None:
        padding_idx = -1
    embedded = torch.nn.functional.embedding(
        inputs, masked_embed_weight, padding_idx, embedding.max_norm, 
        embedding.norm_type, embedding.scale_grad_by_freq, embedding.sparse)
    return embedded


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, drop_ratio):
        # x: [T, N, E]
        # turn off dropout during evaluation or non-drop mode
        if not self.training or not drop_ratio:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1-drop_ratio)
        mask = Variable(m, requires_grad=False)/(1-drop_ratio)
        mask = mask.expand_as(x)
        return mask * x


class LM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # the vocab includes '<blank>'
        vocabSize = config.data.vocabSize
        nemd = config.model.lm.nemd
        nhid = config.model.lm.nhid
        nlayer = config.model.lm.nlayer
        tie_embedding = config.model.lm.tie_embedding
        
        self.embedding = nn.Embedding(vocabSize, nemd)
        self.rnns = nn.LSTM(nemd, nhid, nlayer, batch_first=True)
        self.lm_out = nn.Linear(nhid, vocabSize)
        self.lockdrop = LockedDropout()
        
        if tie_embedding:
            assert nemd==nhid
            self.lm_out.weight = self.embedding.weight
      
    def forward(self, inputs, hidden=None):
        '''
        inputs: [N,T]
        outputs: [N,T,E]
        '''
        embedded = self.embedding(inputs)
#         embedded = embedded_dropout(self.embedding, inputs, 0.2)
#         embedded = self.lockdrop(embedded, 0.2)
        self.rnns.flatten_parameters()
        outputs, hidden = self.rnns(embedded, hidden)
#         outputs = self.lockdrop(outputs, 0.2)
        outputs = self.lm_out(outputs)
        return outputs, hidden
    
    
if __name__ == "__main__":
    ''' To do: pretrain language model on TIMIT '''
    pass



