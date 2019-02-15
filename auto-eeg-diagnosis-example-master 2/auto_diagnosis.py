import logging
import time
from copy import copy
import sys
from dataset import get_all_sorted_file_names_and_labels
import numpy as np
from numpy.random import RandomState
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity

from dataset import DiagnosisSet
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor




def create_set(X, y, inds):
    """
    X list and y nparray
    :return: 
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y,):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.intersect1d(train_inds, test_inds).size == 0
        assert np.intersect1d(valid_inds, test_inds).size == 0
        assert np.array_equal(np.sort(
            np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set


class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.array_equal(np.sort(np.union1d(train_inds, valid_inds)),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set


def run_exp(data_folders,
            n_recordings,
            sensor_types,
            n_chans,
            max_recording_mins,
            sec_to_cut, duration_recording_mins,
            test_recording_mins,
            max_abs_val,
            sampling_freq,
            divisor,
            test_on_eval,
            n_folds, i_test_fold,
            shuffle,
            model_name,
            n_start_chans, n_chan_factor,
            input_time_length, final_conv_length,
            model_constraint,
            init_lr,
            batch_size, max_epochs,cuda,all_file_names):

    #import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))
    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))

    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))
    
    if all_file_names=='standard':
        all_file_names,labels=get_all_sorted_file_names_and_labels('train',['../normal/',
    '../abnormal/'])

    dataset = DiagnosisSet(all_file_names=all_file_names,n_recordings=n_recordings,
                           max_recording_mins=max_recording_mins,
                           preproc_functions=preproc_functions,
                           data_folders=data_folders,
                           train_or_eval='train',
                           sensor_types=sensor_types)
    
    
    #if test_on_eval:
        #if test_recording_mins is None:
            #test_recording_mins = duration_recording_mins
        #test_preproc_functions = copy(preproc_functions)
        #test_preproc_functions[1] = lambda data, fs: (
            #data[:, :int(test_recording_mins * 60 * fs)], fs)
        #test_dataset = DiagnosisSet(n_recordings=n_recordings,
                                #max_recording_mins=None,
                                #preproc_functions=test_preproc_functions,
                                #data_folders=data_folders,
                                #train_or_eval='eval',
                                #sensor_types=sensor_types)
    X,y, y_binary = dataset.load()
 
    #max_shape = np.max([list(x.shape) for x in X],axis=0)
    #print('max',max_shape)
    
    #('SUCCESS.DATA HAS LOADED')
    #assert max_shape[1] == int(duration_recording_mins *sampling_freq * 60)
    #if test_on_eval:
        #test_X, test_y, test_y_binary = test_dataset.load()
        #max_shape = np.max([list(x.shape) for x in test_X],
                           #axis=0)
        #assert max_shape[1] == int(test_recording_mins *
                                   #sampling_freq * 60)
 
    #else:
        #splitter = TrainValidSplitter(n_folds, i_valid_fold=i_test_fold,
                                          #shuffle=shuffle)
        #train_set, valid_set = splitter.split(X, y)
        #test_set = SignalAndTarget(test_X, test_y)
        #del test_X, test_y
    #del X,y # shouldn't be necessary, but just to make sure
    return X,y,y_binary





    # In case you want to recompute predictions for further analysis:
    #exp.model.eval()
    #for setname in ('train', 'valid', 'test'):
        #log.info("Compute predictions for {:s}...".format(
            #setname))
        #dataset = exp.datasets[setname]
        #if config.cuda:
            #preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                      #for b in exp.iterator.get_batches(dataset, shuffle=False)]
        #else:
            #preds_per_batch = [var_to_np(exp.model(np_to_var(b[0])))
                      #for b in exp.iterator.get_batches(dataset, shuffle=False)]
        #preds_per_trial = compute_preds_per_trial(
            #preds_per_batch, dataset,
            #input_time_length=exp.iterator.input_time_length,
            #n_stride=exp.iterator.n_preds_per_input)
        #mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                    #preds_per_trial]
        #mean_preds_per_trial = np.array(mean_preds_per_trial)
