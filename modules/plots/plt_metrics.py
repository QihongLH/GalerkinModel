# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm

def plot_FOM(FOMs, labels, linestyles, colors, FOM_label, y_limits=[]):

    N_FOMs = len(FOMs)
    N_snp = len(FOMs[0])

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(box_aspect=1))

    for i in range(N_FOMs):
        ax.plot(np.arange(1, N_snp + 1), FOMs[i], label=labels[i], linestyle=linestyles[i], color=colors[i])

    ax.set_ylabel(FOM_label)
    ax.set_xlabel('$\Delta N_{snapshot}$')

    ax.set_xlim([1, N_snp + 1])
    ax.set_xticks([1, N_snp + 1])

    if y_limits:
        ax.set_ylim(y_limits)
        ax.set_yticks(y_limits)

    ax.grid()
    ax.legend()
    plt.show()

def plot_Chi(Chi_n):

    m = np.shape(Chi_n)[0]
    n = np.shape(Chi_n)[1]

    Chi_n[ np.where( np.abs(Chi_n) < 1e-5 ) ] = 0
    Chi_n[ np.where( np.abs(Chi_n) > 1e3 ) ] = 1000

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0, n])
    ax.set_ylim([0, m])
    ax.invert_yaxis()

    cp0 = ax.imshow(np.abs(Chi_n), cmap='jet',norm=colors.LogNorm(), aspect=n/m)
    fig.colorbar(cp0, ax=ax, ticks=[1e-4,1,1e2])

    plt.tight_layout()