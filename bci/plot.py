""" Plotting the data and analysis results
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

from . import model

def pvalues(time, X, y, pos, ch2id):
    ch_Cz = ch2id["Cz"]
    ch_Pz = ch2id["Pz"]

    dpos = (pos - pos[ch_Cz])
    r    = ((dpos**2).sum(axis=1))**0.5

    color = list(sns.color_palette()[0])

    _, pi = scipy.stats.ttest_ind(X[y == 0], X[y == 1], axis=0)
    p = np.prod(pi, axis=0)

    idc = np.argsort(p)

    labels = {ch_Cz : 'Cz', ch_Pz : 'Pz'}

    for i, pii in enumerate(pi):
        plt.plot(time, -np.log(pii), color=color, alpha=1-r[i]/r.max(), label=labels.get(i, None))

    plt.xlabel("Time [ms]")
    plt.ylabel("-$\log \ p$")
    plt.title("$p$ based on radius from Cz")
    plt.legend()
    sns.despine()
    plt.show()

def uncertainty(t, data, ax=None, c=None):

    if ax is None: ax = plt.gca()

    mu_train  = data.mean(axis=0)
    std_train = data.std(axis=0)

    ax.plot(t, mu_train)
    ax.fill_between(t, mu_train - 1 * std_train, mu_train + 1 * std_train, alpha=.5)


def spectrogram(F_ff, F_tt, Sxx, y, ch, fname):

    sns.set_context('paper', font_scale=1.5)

    def show_cbar(obj, ax):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(obj,cax=cax,orientation='vertical')

    _, p_f = scipy.stats.ttest_ind(Sxx[y == 0], Sxx[y == 1], axis=0)

    fig, axes = plt.subplots(1,3,figsize=(15,3), sharey=True, sharex=True)

    data = [Sxx[y==1].mean(axis=0)[ch],
            Sxx[y==0].mean(axis=0)[ch],
            -np.log10(p_f)[ch]]
    titles = ["(A) Error", "(B) No Error", "(C) p-value on difference"]
    vmaxs = [5,5,None]

    for ax, S, title, vmax in zip(axes, data, titles, vmaxs):
        mesh = ax.pcolormesh(F_tt, F_ff, S, vmin=0, vmax=vmax)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title(title)
        ax.grid("off")
        show_cbar(mesh, ax)

    plt.tight_layout()
    plt.savefig('report/fig/{}.pdf'.format(fname), bbox_inches='tight')
    plt.show()

def overview(X, y, t, ch_Cz, ch_name, ch_label, pos, fname):

    p, pi, chs, ids = model.compute_p_values(X, y, t)

    sns.set_context('paper', font_scale=2.5)
    fig, axes = plt.subplots(3,5,figsize=(20,10))

    axes[0,0].set_title('Error')
    axes[0,1].set_title('No Error')
    axes[0,4].set_title("p-values")

    for ax, idx, ch in zip(axes, ids, chs):

        mne.viz.plot_topomap(X[y == 1,:,idx].mean(axis=0), pos, axes=ax[0], show=False)
        mne.viz.plot_topomap(X[y == 0,:,idx].mean(axis=0), pos, axes=ax[1], show=False)

        ax[2].plot([t[idx]]*2,[-10,10],c='black',alpha=0.5)
        ax[3].plot([t[idx]]*2,[-10,10],c='black',alpha=0.5)

        ax[2].fill_between(t, np.zeros_like(pi[ch])-10, 20*(-np.log10(pi[ch]) > 50)-10,                           interpolate=False, color="black", alpha=0.5)

        ax[3].fill_between(t, np.zeros_like(pi[ch_Cz])-10, 20*(-np.log10(pi[ch_Cz]) > 5)-10,                           interpolate=False, color="black", alpha=0.5)

        uncertainty(t, X[y == 0,ch], c='blue',      ax=ax[2])
        uncertainty(t, X[y == 1,ch], c='orange',    ax=ax[2])
        uncertainty(t, X[y == 0,ch_Cz], c='blue',   ax=ax[3])
        uncertainty(t, X[y == 1,ch_Cz], c='orange', ax=ax[3])

        ax[2].set_title('{}'.format(ch_label[ch]))
        ax[3].set_title('{}'.format(ch_label[ch_Cz]))

        ax[2].set_ylim(-10,10)
        ax[3].set_ylim(-10,10)

        sns.despine(ax=ax[2])
        sns.despine(ax=ax[3])

        mne.viz.plot_topomap(np.log(-np.log(pi))[:,idx], pos, axes=ax[4], show=False,
                             vmin=0, vmax=np.log(100))

        ax[0].set_ylabel("t = {:.1f} ms".format(t[idx]))

    plt.tight_layout()
    plt.savefig('report/fig/{}.pdf'.format(fname), bbox_inches='tight')
    plt.show()

def scatter_features(features, y):
    fig, axes = plt.subplots(3,3,figsize=(10,10))
    for i, axes_ in enumerate(axes):
        for j, ax in enumerate(axes_):
            ax.axis('off')
            if i == j: continue
            ax.scatter(features[:,i], features[:,j], c=y, marker='x', cmap="Set1")

    plt.show()

def features(f1, y1, f2, y2):
    sns.set_context("paper", font_scale=1.5)
    fig, ax = plt.subplots(3,3,figsize=(10,5),sharex=True,sharey=True)
    ax = ax.flatten()
    for i in range(f1.shape[1]):
        sns.despine()

        sns.kdeplot(f1[y1==0,i], ax=ax[i], color="blue",label="$\Omega_s$ No Err", legend=False)
        sns.kdeplot(f1[y1==1,i], ax=ax[i], color="red",label="$\Omega_s$ Err", legend=False)

        sns.kdeplot(f2[y2==0,i], ax=ax[i], color="blue", ls="--",label="$\Omega_t$ No Err",legend=False)
        sns.kdeplot(f2[y2==1,i], ax=ax[i], color="red", ls="--",label="$\Omega_t$ Err",legend=False)

    ax[i].legend()

    plt.tight_layout()
    plt.show()

def disagreement(time, X,y1,y2,ch):

    sns.set_context("paper", font_scale=1.5)
    fig, axes = plt.subplots(2,2,figsize=(10,5),sharex=True,sharey=True)
    sns.despine()

    data = np.zeros((2,2)+X.shape[1:])
    for i in range(len(y1)):
        data[y1[i]*1,y2[i]*1] += X[i,ch]

    data /= len(y1)

    for i in range(len(y1)):
        axes[y1[i]*1,y2[i]*1].plot(time,data[y1[i]*1,y2[i]*1,ch])
    plt.tight_layout()
