""" Model selection and data processing
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

import sklearn.model_selection
import sklearn.svm
import sklearn.discriminant_analysis
from sklearn.metrics import confusion_matrix
import pandas as pd

### ----------------------- Feature Extraction  ----------------------- ###

def augment_data(X, y):
    return X, y
    X_new = []
    y_new = []
    for x,y in zip(X, y):
        for dt in range(-10,10,5):
            X_new.append(np.roll(x, dt, axis=-1))
            y_new.append(y)

    return np.stack(X_new, axis=0), np.stack(y_new, axis=0)

def compute_spectogram(X):
    F_ff, F_tt, Sxx = scipy.signal.spectrogram(X, fs=256, nperseg=128,noverlap=100, axis=-1)
    F_tt -= 3
    idf = F_ff < 20
    idt = (F_tt > -.5) & (F_tt < 1.)
    return F_ff[idf], F_tt[idt], Sxx[...,idf,:][..., idt]

def compute_p_values(X, y, t, n_channels=1):
    _, pi = scipy.stats.ttest_ind(X[y == 0], X[y == 1], axis=0)
    p = np.prod(pi, axis=0)

    # selector for specific time points
    select = lambda a,b : np.arange(len(p))[(t > a) & (t < b)][np.argmax(-np.log(1e-200+p)[(t > a) & (t < b)])]

    ids = [ select(180,210),
            select(250,350),
            select(350,450) ]

    chs = np.concatenate( [(np.log(pi)[:,i]).argsort()[:n_channels] for i in ids] )
    ids = np.concatenate( [np.array([i]*n_channels) for i in ids])

    return p, pi, chs, ids

def get_features(X, chs, ids):
    return np.stack([ abs(X[:,ch,i-10:i+10]).max(axis=-1) for ch,i in zip(chs, ids)],\
                    axis=-1)


def get_more_features(X, chs, ids):
    return np.concatenate([ abs(X[:,ch,i-10:i+10]) for ch in chs for i in ids],\
                    axis=-1)


### ----------------------- Domain Adaptation -------------------- ###

def normalize(y, axis=None):
    ymax = y.max(axis=axis, keepdims=True)
    ymin = y.min(axis=axis, keepdims=True)

    return (y - ymin) / (ymax - ymin)

def optimal_transport(Xs, Xt, ys=None, norm=True):
    """ Apply Optimal Transport with Sinkhorn metric """

    if normalize:
        Xs = normalize(Xs.copy(), axis=0)
        Xt = normalize(Xt.copy(), axis=0)

    if ys is None:
        ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
        ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
    else:
        ot_lpl1 = ot.da.SinkhornLpl1Transport(reg_e=1e-1, reg_cl=1e0)
        ot_lpl1.fit(Xs=Xs, ys=ys, Xt=Xt)
        transp_Xs_sinkhorn = ot_lpl1.transform(Xs=Xs)

    return transp_Xs_sinkhorn, Xt

### ----------------------- Custom Models  ----------------------- ###

class CorrelationEstimator():
    """ Refine an existing partition by correlation estimation
    """

    def __init__(self, channel=slice(0, None), idx=slice(0, None),\
                 eps=0.2, func_='softmax', threshold=0.7):

        self.eps     = eps
        self.idx     = idx
        self.channel = channel
        self.threshold = threshold

        self.func = lambda x,y : np.exp(x) / (np.exp(x) + np.exp(y)) if func_ == 'softmax' else func_

    def fit(self, X, y):
        # templates for classes
        self.k_noerror = X[y == 0].mean(axis=0,keepdims=True)
        self.k_error   = X[y == 1].mean(axis=0,keepdims=True)

        self.k_noerror -= self.k_noerror.mean(axis=-1, keepdims=True)
        self.k_error -= self.k_noerror.mean(axis=-1, keepdims=True)

    def predict(self, X, y):

        # estimate classes
        id_no_error   = (y < self.eps)
        id_error      = (y > 1 - self.eps)
        id_unsure     = (abs(.5-y) < 0.5-self.eps)

        if id_unsure.sum() == 0: return id_error

        x             = X[id_unsure].copy()
        x             = x - x.mean(axis=-1, keepdims=True)

        score_error   = (x*self.k_error)  [:,self.channel][:,:,self.idx].mean(axis=(1,2))
        score_noerror = (x*self.k_noerror)[:,self.channel][:,:,self.idx].mean(axis=(1,2))
        p_comp        = self.func(score_error,score_noerror)

        id_error[id_unsure] = p_comp > 0.8 #self.threshold

        return id_error

### ----------------------- Evaluation  ----------------------- ###

def build_lda():
    return [[sklearn.discriminant_analysis.LinearDiscriminantAnalysis(), "LDA"]]

def build_svms():
    return [[sklearn.svm.LinearSVC(C=i, class_weight='balanced'), 'LinSVC_C {:.3e}'.format(i)] for i in [1]] +\
            [[sklearn.svm.SVC(C=i, class_weight='balanced'), 'SVC_C {:.3e}'.format(i)] for i in np.logspace(-6,5)]

def build_rfos():
    return [[sklearn.ensemble.RandomForestClassifier(n_estimators=i), 'RFO_n {:.3e}'.format(i)] for i in range(2,20,2)]

def build_classifiers():
    return build_svms() + build_rfos() + build_lda()


def run_experiments(Xt, yt, features, n_splits=10):
    folds = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)

    methods = build_classifiers()
    acc = pd.DataFrame(columns=[n for _,n in methods], index=range(n_splits))
    f1  = pd.DataFrame(columns=[n for _,n in methods], index=range(n_splits))
    models  = {}

    for i_split, (idct, idcv) in enumerate(folds.split(Xt, yt)):

        X, y   = Xt[idct], yt[idct]
        Xv, yv = Xt[idcv], yt[idcv]

        z  = features[idct]
        zv = features[idcv]

        methods = build_classifiers()

        for model, name in methods:
            model.fit(z, y)

            pred  = model.predict(z)
            predv = model.predict(zv)

            acc_train = (pred  == y).mean()
            acc_val   = (predv == yv).mean()

            acc[name].loc[i_split] = acc_val
            f1[name].loc[i_split]  = acc_val #sklearn.metrics.f1_score(yv, predv)

            if not name in models.keys(): models[name] = []
            models[name] += [model]
    return acc, f1, models

def get_ensemble():
    return [(sklearn.svm.LinearSVC(C=1, class_weight='balanced'), 'LinSVC'),
            (sklearn.ensemble.RandomForestClassifier(n_estimators=2), 'RFO'),
            (sklearn.discriminant_analysis.LinearDiscriminantAnalysis(), 'LDA')]


#def get_ensemble():
    #return [(sklearn.svm.LinearSVC(C=100, class_weight='balanced'), 'LinSVC')]

def train_final_models(Xt, yt, features, n_splits=10, n_runs=10, prefix=""):

    acc = pd.DataFrame(columns=[prefix+n for n in ['Ensemble', 'Final']], index=range(n_splits))
    f1  = pd.DataFrame(columns=[prefix+n for n in ['Ensemble', 'Final']], index=range(n_splits))
    acc[:] = 0
    f1[:] = 0
    models  = []

    for i_run in range(n_runs):
        folds = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=False)

        for i_split, (idct, idcv) in enumerate(folds.split(Xt, yt)):

            X, y   = Xt[idct], yt[idct]
            Xv, yv = Xt[idcv], yt[idcv]

            z  = features[idct]
            zv = features[idcv]

            resolver = CorrelationEstimator(eps=0.2,idx=slice(174,242), channel=[10, 19])
            resolver.fit(X,y)

            methods = get_ensemble() + get_ensemble() + get_ensemble() + get_ensemble()
            for model, name in methods:
                idc = np.arange(len(z))
                idc = np.random.choice(idc, replace=False, size=int(len(idc)*0.3))
                model.fit(z[idc], y[idc])

            predict = lambda x: sum([e.predict(x) for e,_ in methods]) / len(methods)

            pred  = predict(z)
            predv = predict(zv)

            acc_train = ((pred > .35)  == y).mean()
            acc_val   = ((predv > .35) == yv).mean()
            acc[prefix+'Ensemble'].loc[i_split] += acc_val
            f1[prefix+'Ensemble'].loc[i_split]  += sklearn.metrics.f1_score(yv, predv > .5)

            pred  = resolver.predict(X, pred)
            predv = resolver.predict(Xv, predv)

            acc_train = (pred  == y).mean()
            acc_val   = (predv == yv).mean()
            acc[prefix+'Final'].loc[i_split] += acc_val
            f1[prefix+'Final'].loc[i_split]  += sklearn.metrics.f1_score(yv, predv)

            models += methods

    predict = lambda x: sum([e.predict(x) for e,_ in models]) / len(models)
    return acc / n_runs, f1 / n_runs, models, predict

### ----------------------- Evaluation  ----------------------- ###

def build_ensemble(results, models, n_models=5, eps=0):
    best = results.mean().idxmax()
    best, results[best].mean(), results[best].std()

    rank = (results.mean()).sort_values(ascending=False)[0:n_models]
    best_names = list(rank.index)

    ensemble = [m for n in best_names for m in models[n]]
    pred = lambda x: sum([e.predict(x) for e in ensemble]) / len(ensemble)

    def predict(x):
        p_test = pred(x)
        y_test = np.zeros_like(p_test) + 2
        y_test[p_test > .5+eps] = 1
        y_test[p_test < .5-eps] = 0

        return y_test

    return predict, best_names
