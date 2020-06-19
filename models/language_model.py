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
        rnns = [nn.LSTM(
            nemd if l==0 else nhid, nhid if l!=nlayer-1 else nemd, 
            batch_first=True) for l in range(nlayer)]
        self.rnns = nn.ModuleList(rnns)
        self.out = nn.Linear(nemd, vocabSize)
        
        if tie_embedding:
            self.out.weight = self.embedding.weight
      
    def forward(self, inputs):
        '''
        inputs: [N,T]
        outputs: [N,T,E]
        '''
        [rnn.flatten_parameters() for rnn in self.rnns]
        embedded = self.embedding(inputs)
        outputs,_ = self.rnns(embedded)
        outputs = self.out(outputs)
        return outputs
    
    
if __name__ == "__main__":
    ''' To do: pretrain language model on TIMIT '''
    pass



