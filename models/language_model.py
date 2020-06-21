import torch
import torch.nn as nn


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
        
        if tie_embedding:
            assert nemd==nhid
            self.lm_out.weight = self.embedding.weight
      
    def forward(self, inputs, hidden=None):
        '''
        inputs: [N,T]
        outputs: [N,T,E]
        '''
        embedded = self.embedding(inputs)
        self.rnns.flatten_parameters()
        outputs, hidden = self.rnns(embedded, hidden)
        outputs = self.lm_out(outputs)
        return outputs, hidden
    
    
if __name__ == "__main__":
    ''' To do: pretrain language model on TIMIT '''
    pass



