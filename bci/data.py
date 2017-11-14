""" Data and metadata loading
"""

__author__ = 'Steffen Schneider'
__email__  = 'steffen.schneider@tum.de'


import numpy as np

import scipy
import scipy.io
import scipy.signal

import matplotlib.pyplot as plt
import seaborn as sns

import mne
import ot
import sklearn

from bci import plot
import tools

def load_dataset(fname):
    fmt = lambda x: x.transpose((2,0,1))

    dataset = scipy.io.loadmat(fname)
    X_train_0 = fmt(dataset['X_train_0'])
    X_train_1 = fmt(dataset['X_train_1'])
    X_val     = fmt(dataset['X_val'])

    X_train = np.concatenate([X_train_0, X_train_1], axis=0)
    y_train = np.concatenate([np.zeros(len(X_train_0)), np.ones(len(X_train_1))])

    return (X_train, y_train), (X_val, None)

def load_calibration(fname):
    fname = 'data/calibration.set'
    dataset = scipy.io.loadmat(fname)
    ch_label = np.concatenate(dataset['EEG']['chanlocs'][0,0]['labels'][0])
    xx = np.concatenate(dataset['EEG']['chanlocs'][0,0]['X'][0])
    yy = np.concatenate(dataset['EEG']['chanlocs'][0,0]['Y'][0])
    zz = np.concatenate(dataset['EEG']['chanlocs'][0,0]['Z'][0])

    pos = np.concatenate([xx,yy], axis=1)
    ch2id = {name : i for i, name in enumerate(ch_label)}

    return pos, ch2id, ch_label
