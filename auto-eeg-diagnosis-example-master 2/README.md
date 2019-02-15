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
There are two folders in the main directory, train and eval. Inside each folder are two subfolders, abnormal and normal. Each contain the respective EEG signals.
