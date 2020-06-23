# End-to-End Automatic Speech Recognition
This repository contrains implementations of end-to-end ASR system by LAS, CTC(w/o attention), and transducer(w/o attention).


## Dependencies
- torch >= 1.5.1
- torchtext >= 0.6.0
- torchaudio
- warp-rnnt


## Phoneme Error Rate on TIMIT

|   Model                |   Epoch   |  Loss (train/dev) |  Per (train/dev)  |
| :------------------:   |:---------:| :----------------:|:-----------------:|
| CTC                    |      1.8257/0.5512/100/6h/No      |     1.2453/0.3761/300/1d+/No   |
| Transducer             |      12.442/0.3048/38/1.5h/Yes    |                                |
| Pretrained Transducer  |                                   |                                |
| LAS                    |                                   |                                |



## Note
1. Smaller vocabulary (due to phoneme mapping<sup>[7](#Reference)</sup>) improves performance.
2. VGG Feature extractor<sup>[7](#Reference)</sup> (ResNet even better) helps model to converge fast.
3. Transducer converges faster and generalizes better than ctc.
4. Weight noise<sup>[8](#Reference)</sup> is a useful regularizer for RNN/LSTM.
5. Batch normalization helps model to converge fast.
 


## To do
1. pretrained LM and CTC
2. beam search
3. LAS
4. hybrid 



## Reference
1. A Comparison of Sequence-to-Sequence Models for Speech Recognition [[Ref](https://www.isca-speech.org/archive/Interspeech_2017/abstracts/0233.html)]
2. Deep Learning for Human Language Processing (2020,Spring) [[Ref](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html)]
3. Alexander-H-Liu/End-to-end-ASR-Pytorch [[Ref](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)]
4. Open Source Korean End-to-end Automatic Speech Recognition [[Ref](https://github.com/sooftware/KoSpeech)]
5. LANGUAGE TRANSLATION WITH TORCHTEXT [[Ref](https://github.com/pytorch/tutorials/blob/master/beginner_source/torchtext_translation_tutorial.py)]
6. End-to-end automatic speech recognition system implemented in TensorFlow [[Ref](https://github.com/zzw922cn/Automatic_Speech_Recognition)]
7. Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM [[Ref](https://arxiv.org/pdf/1706.02737.pdf)]
8. Speech Recognition with Deep Recurrent Neural Networks [[Ref](https://arxiv.org/abs/1303.5778)]



## Acknowledge
- Thanks to [warp-rnnt](https://github.com/1ytic/warp-rnnt/tree/master/pytorch_binding), a PyTorch bindings for CUDA-Warp RNN-Transducer. Note that it is better installed from source code.
- Thanks to [warp-transducer](https://github.com/HawkAaron/warp-transducer/tree/master/pytorch_binding), a more general implementation of RNN transducer.  Carefully set the environment variables as [refered here](https://github.com/HawkAaron/warp-transducer/issues/15) before run ```python setup.py install``` .

