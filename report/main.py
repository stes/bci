import numpy as np
import sklearn
import scipy
import scipy.io
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import sklearn.ensemble


# dataset is in eeg lab format CHANNEL x TIMESTEPS X EPOCH

def plot_uncertainty(t, data, ax=None, c=None):

    if ax is None: ax = plt.gca()

    mu_train  = data.mean(axis=0)
    std_train = data.std(axis=0)

    ax.plot(t, mu_train)
    ax.fill_between(t, mu_train - 1 * std_train, mu_train + 1 * std_train, alpha=.5)

def load_data(fname):
    fmt = lambda x: x.transpose((2,0,1))

    dataset = scipy.io.loadmat(fname)
    X_train_0 = fmt(dataset['X_train_0'])
    X_train_1 = fmt(dataset['X_train_1'])
    X_val     = fmt(dataset['X_val'])

    X_train = np.concatenate([X_train_0, X_train_1], axis=0)
    y_train = np.concatenate([np.zeros(len(X_train_0)), np.ones(len(X_train_1))])

    return (X_train, y_train), (X_val, None)

def augment_data(X, y):
    return X, y
    X_new = []
    y_new = []
    for x,y in zip(X, y):
        for dt in range(-10,10,5):
            X_new.append(np.roll(x, dt, axis=-1))
            y_new.append(y)

    return np.stack(X_new, axis=0), np.stack(y_new, axis=0)

def train():
    folds = sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True)
    names      = ['svm1','svm10','svm100','svml1', 'rfo']
    pred_train = {n : np.zeros_like(y_train) - 1 for n in names}
    pred_valid = {n : np.zeros(X_val.shape[0]) - 1 for n in names}

    for idct, idcv in folds.split(X_train, y_train):

        X, y = augment_data(X_train[idct], y_train[idct])
        Xv, yv = X_train[idcv], y_train[idcv]

        # determine statistical significance
        _,pvalues = scipy.stats.ttest_ind(X[y == 0],
                                          X[y == 1], axis=0)
        i,j = np.where(pvalues[:,:] < 0.001)
        z   = X[:,i,j]

        pca          = PCA(whiten=True, n_components=10)
        z_pca        = pca.fit_transform(z)

        svm = sklearn.svm.LinearSVC(C=1)
        svm10 = sklearn.svm.LinearSVC(C=10)
        svm100 = sklearn.svm.LinearSVC(C=100)
        svml1  = sklearn.svm.LinearSVC(C=0.01, penalty="l1", dual=False)
        rfo = sklearn.ensemble.RandomForestClassifier()

        methods = [svm,svm10,svm100,svml1,rfo]

        for name, model in zip(names, methods):
            model.fit(z_pca, y)

            pred = lambda x : model.predict(pca.transform(x[:,i,j]))

            pred_train[name][idcv] = pred(Xv)
            pred_valid[name]       = pred(X_val)

            acc_train = (pred(X) == y).mean()
            acc_val   = (pred_train[name][idcv] == yv).mean()

            print("="*80)
            print(name)
            print('Train: {:.3f} Val: {:.3f}'.format(acc_train, acc_val))
            print(confusion_matrix(pred(X), y))
            print(confusion_matrix(pred(Xv), yv))
