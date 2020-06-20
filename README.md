# End-to-End Automatic Speech Recognition
This repository contrains implementations of end-to-end ASR system by LAS, CTC(w/o attention), and transducer(w/o attention).


## Dependencies
- torch >= 1.5.1
- torchtext >= 0.6.0
- torchaudio
- warp-rnnt


## Phoneme Error Rate on TIMIT

|   Model                |   Loss/PER/epochs/time/converge   |  Loss/PER/epochs/time/converge | 
| :------------------:   |   :----------------------------:  | :-----------------------------:|
| CTC (w/o PM)           |      2.3436/0.6812/100/6h/No      |                                |
| CTC                    |      1.8257/0.5512/100/6h/No      |     1.2453/0.3761/300/1d+/No   |
| Transducer             |                                   |                                |

**PM:**  Phoneme mapping from 61 to 39 target phonemes ([Ref](https://github.com/zzw922cn/Automatic_Speech_Recognition)).


## Note
1. Smaller vocabulary model (w/ PM) converges faster and generalizes better than larger vocabulary model (w/o PM).
2. Generalization: VGG2 feature extractor > VGG4 feature extractor > No feature extractor.
3. Training gets slower as epoch iterates. (cpu util?)

 

## To do
1. weight noise
2. transducer
3. LAS
4. hybrid 


## Reference

- A Comparison of Sequence-to-Sequence Models for Speech Recognition [[Ref](https://www.isca-speech.org/archive/Interspeech_2017/abstracts/0233.html)]
- Deep Learning for Human Language Processing (2020,Spring) [[Ref](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html)]
- Alexander-H-Liu/End-to-end-ASR-Pytorch [[Ref](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)]
- Open Source Korean End-to-end Automatic Speech Recognition [[Ref](https://github.com/sooftware/KoSpeech)]
- LANGUAGE TRANSLATION WITH TORCHTEXT [[Ref](https://github.com/pytorch/tutorials/blob/master/beginner_source/torchtext_translation_tutorial.py)]


## Acknowledge

- Thanks to [warp-rnnt](https://github.com/1ytic/warp-rnnt/tree/master/pytorch_binding), a PyTorch bindings for CUDA-Warp RNN-Transducer. Note that it is better installed from source code.
- Thanks to [warp-transducer](https://github.com/HawkAaron/warp-transducer/tree/master/pytorch_binding), a more general implementation of RNN transducer.  Carefully set the environment variables as [refered here](https://github.com/HawkAaron/warp-transducer/issues/15) before run ```python setup.py instal``` .

