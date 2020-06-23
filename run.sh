# @github{
#   title = {End-to-End Automatic Speech Recognition},
#   author = {Xie Liang},
#   link = {https://github.com/xieliang555/Automatic-Speech-Recognition},
#   year = {2020}
# }


DEVICE=0
SEED=1111

cd trainer

python ctc_train.py --device $DEVICE --seed $SEED
