# auto-eeg-diagnosis-example

## Requirements
1. Install https://robintibor.github.io/braindecode/ 


## Run
1. Run 'python kerasgetmodel.py'


## Model Architecture
1. The encoder architecture for the eeg is implemented from https://arxiv.org/abs/1807.03402.
2. The decoder architecture is a multi-layered CNN with spatial attention.

## Modifications
You can modify the model architecture in kerasgetmodel.py

## Dataset folders--same as temple eeg data
---train
-----abnormal
-----normal
---eval
-----abnormal
-----normal

```
@INPROCEEDINGS{schirrmreegdiag2017,
  author={R. Schirrmeister and L. Gemein and K. Eggensperger and F. Hutter and T. Ball},
  booktitle={2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  title={Deep learning with convolutional neural networks for decoding and visualization of EEG pathology},
  year={2017},
  volume={},
  number={},
  pages={1-7},
  ISSN={},
  month={Dec},}
```
