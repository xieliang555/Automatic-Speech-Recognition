# End-to-End Automatic Speech Recognition
This repository contrains implementations of end-to-end ASR system by LAS, CTC(w/o attention), and transducer(w/o attention).


## Dependencies
- torch >= 1.5.1
- torchtext >= 0.6.0
- torchaudio
- warp-rnnt


## Phoneme Error Rate on TIMIT

|   Model                |  train/dev loss |  train/dev per  |   Epoch   |
| :------------------:   |:---------------:| :--------------:|:---------:|
| CTC                    |   0.64/1.03     |   0.20/0.315    |   178     |
| Transducer             |   12.0/-        |     -/0.2662    |   13      |
| Pretrained Transducer  |   0.7/-         |     -/0.2670    |   195     |
| LAS                    |                 |                 |           |


language model
train/dev loss: 2.68/2.80
train/dev ppl: 14.5/16.49
epoch: 292


## Note
1. Smaller vocabulary (due to phoneme mapping<sup>[6](#Reference)</sup>) improves performance.
2. VGG Feature extractor<sup>[7](#Reference)</sup> (ResNet even better) helps model to converge fast.
3. Transducer converges faster and generalizes better than ctc.
4. Weight noise<sup>[8](#Reference)</sup> is a useful regularizer for RNN/LSTM.
5. Batch normalization helps model to converge fast.
 


## To do
1. pretrained transducer
2. LAS
3. beam search
4. hybrid 
5. add visualize script plot.py



## Reference
1. A Comparison of Sequence-to-Sequence Models for Speech Recognition [[Ref](https://www.isca-speech.org/archive/Interspeech_2017/abstracts/0233.html)]
2. Deep Learning for Human Language Processing (2020,Spring) [[Ref](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html)]
3. Alexander-H-Liu/End-to-end-ASR-Pytorch [[Ref](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)]
4. Open Source Korean End-to-end Automatic Speech Recognition [[Ref](https://github.com/sooftware/KoSpeech)]
5. Language Translation With TorchText [[Ref](https://github.com/pytorch/tutorials/blob/master/beginner_source/torchtext_translation_tutorial.py)]
6. End-to-end automatic speech recognition system implemented in TensorFlow [[Ref](https://github.com/zzw922cn/Automatic_Speech_Recognition)]
7. Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM [[Ref](https://arxiv.org/pdf/1706.02737.pdf)]
8. Speech Recognition with Deep Recurrent Neural Networks [[Ref](https://arxiv.org/abs/1303.5778)]
9. pretrained embedding [[Ref](https://github.com/pytorch/examples/blob/master/snli/train.py)]



## Acknowledge
- Thanks to [warp-rnnt](https://github.com/1ytic/warp-rnnt/tree/master/pytorch_binding), a PyTorch bindings for CUDA-Warp RNN-Transducer. Note that it is better installed from source code.
- Thanks to [warp-transducer](https://github.com/HawkAaron/warp-transducer/tree/master/pytorch_binding), a more general implementation of RNN transducer.  Carefully set the environment variables as [refered here](https://github.com/HawkAaron/warp-transducer/issues/15) before run ```python setup.py install``` .

