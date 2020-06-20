import torch
import torchaudio
from torch.utils.data import Dataset

import os
from pathlib import Path
import soundfile as sf


class TIMIT(Dataset):
    def __init__(self, root, mode):
        '''
        description:
            only using core test set
            mapping 61 phonemes to 39 phonemes
        '''
        core_test_list = ['MDAB0', 'MWBT0', 'FELC0',
                          'MTAS1', 'MWEW0', 'FPAS0',
                          'MJMP0', 'MLNT0', 'FPKT0',
                          'MLLL0', 'MTLS0', 'FJLM0',
                          'MBPM0', 'MKLT0', 'FNLP0',
                          'MCMJ0', 'MJDH0', 'FMGD0',
                          'MGRT0', 'MNJM0', 'FDHC0',
                          'MJLN0', 'MPAM0', 'FMLD0']
        
        self.mapping = {'ih':'ix', 'ah':'ax', 'ax-h':'ax',
                        'ux':'uw', 'aa':'ao', 'axr':'er',
                        'el':'l', 'em':'m', 'en':'n', 'nx':'n',
                        'eng':'ng', 'sh':'zh', 'hv':'hh', 'bcl':'h#',
                        'pcl':'h#', 'dcl':'h#', 'tcl':'h#', 'gcl':'h#',
                        'kcl':'h#',   'q':'h#', 'epi':'h#', 'pau':'h#'}
        
        self.mode = mode
        root = Path(os.path.join(root, 'TIMIT/data/lisa/data/timit/raw/TIMIT'))
        
        if self.mode == 'train':
            self.wav_paths = sorted(
                [str(x) for x in root.glob('TRAIN/**/*.WAV')])
            txt_paths = sorted(
                [str(x) for x in root.glob('TRAIN/**/*.PHN')])
        else:
            wav_paths = sorted([str(x) for x in root.glob(
                'TEST/**/*.WAV') if 'SA1' not in str(x) and 'SA2' not in str(x)])
            self.wav_paths = [
                p for p in wav_paths if p.split('/')[13] in core_test_list]
            txt_paths = sorted([str(x) for x in root.glob(
                'TEST/**/*.PHN') if 'SA1' not in str(x) and 'SA2' not in str(x)])
            txt_paths = [
                p for p in txt_paths if p.split('/')[13] in core_test_list]
            
        self.waveform =[sf.read(p)[0] for p in self.wav_paths]
        self.txt = []
        for p in txt_paths:
            with open(p) as f:
                data = f.read().split()
                data = [d for d in data if d.isalpha()]
                self.txt.append(data)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        waveform = torch.from_numpy(self.waveform[idx]).view(1, -1).float()
        # get mfcc/fbank feature
        feature = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=40)
        # add deltas
        d1 = torchaudio.functional.compute_deltas(feature)
        d2 = torchaudio.functional.compute_deltas(d1)
        feature = torch.cat([feature, d1, d2], dim=-1)
        # CMVN normalization
        mean = feature.mean(0, keepdim=True)
        std = feature.std(0, keepdim=True)
        feature = (feature-mean)/(std + 1e-10)
        
        txt = self.txt[idx]
        txt = [self.mapping.get(t,t) for t in txt]
        return feature, txt