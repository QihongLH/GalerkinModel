import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modules.plots.plt_config import plot_body

def plot_energy_POD(Sigma):

    cum_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
    energy = Sigma**2 / np.sum(Sigma**2)

    nr = len(energy)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(box_aspect=1))

    ax[0].semilogx(np.arange(1, nr + 1), energy) # or loglog
    ax[0].set_title('Energy')
    ax[0].set_xlabel('$n_r$')
    ax[0].axis([1, nr, 0, 1])
    ax[0].grid()

    ax[1].semilogx(np.arange(1, nr + 1), cum_energy) # or loglog
    ax[1].set_title('Cumulative energy')
    ax[1].set_xlabel('$n_r$')
    ax[1].axis([1, nr, 0, 1])
    ax[1].grid()

    plt.tight_layout()
    plt.show()

def plot_psi(t, t_IC, X_true, X_GP, X_interp, X_IC, nr, nf):

    if nr > 5:
        nc = 5
        nr = int(np.ceil(nr/5))
    else:
        nc = nr
        nr = 1

    fig, ax = plt.subplots(nr, nc, subplot_kw=dict(box_aspect=1))

    if nr != 1 and nc != 1:
        for i in range(nr):
            for j in range(nc):
                ax[i, j].locator_params(axis='both', nbins=3)

                ax[i, j].plot(t, X_true[:,i * nc + j], 'k-', label='True')
                ax[i, j].plot(t, X_interp[:,i * nc + j], 'g--', label='Interp')
                ax[i, j].plot(t, X_GP[:, i * nc + j], 'b--', label='GP')
                ax[i, j].plot(t_IC, X_IC[:, i * nc + j], 'ro', label='ICs')

                if i == nr-1:
                    ax[i, j].set_xlabel(r'$t/\tau$')
                    ax[i, j].set_xticks([t[0], t[nf]])
                else:
                    ax[i, j].set_xticks([])

                ax[i, j].set_ylabel(r'$a_{'+str(i * nc + j + 1)+'}$')

                ax[i, j].set_xlim([t[0], t[nf]])

    elif nr == 1 and nc != 1:
        for j in range(nc):
            ax[j].locator_params(axis='both', nbins=3)

            ax[j].plot(t, X_true[:, j], 'r-', label='True')
            ax[j].plot(t, X_interp[:, j], 'g--', label='Interp')
            ax[j].plot(t, X_GP[:, j], 'b--', label='GP')
            ax[j].plot(t_IC, X_IC[:, j], 'ro', label='IC')

            ax[j].set_ylabel(r'$a_{'+str(j+1)+'}$')
            ax[j].set_xlabel(r'$t/\tau$')

            ax[j].set_xlim([t[0], t[nf]])
            ax[j].set_xticks([t[0], t[nf]])

    else:
        ax.locator_params(axis='both', nbins=3)

        ax.plot(t, X_true[:,0], 'r-', label='True')
        ax.plot(t, X_interp[:, 0], 'g--', label='Interp')
        ax.plot(t, X_GP[:, 0], 'b--', label='GP')
        ax.plot(t_IC, X_IC[:, 0], 'ro', label='IC')

        ax.set_xlabel(r'$t/\tau$')
        ax.set_ylabel(r'$a_{1}$')

        ax.set_xlim([t[0], t[nf]])
        ax.set_xticks([t[0], t[nf]])

    if nr != 1 and nc != 1:
        ax[-1, -1].legend(labelcolor='black')
    elif nr == 1 and nc != 1:
        ax[-1].legend(labelcolor='black')
    else:
        ax.legend(labelcolor='black')

    plt.tight_layout()
    plt.show()

def video_psi_sigma_phi(grid, t, Phi, Sigma, PsiT, nr, path_out, flag_flow = 'FP', limits = [-0.5, 0.5], limitsm = [-0.01, 0.01]):

    # Parameters
    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m, n = np.shape(X)
    nt = np.shape(PsiT)[1]
    nrows, ncols = 3, nr

    titles = [r'$\phi_{r}$', r'$\psi_{r}$', r'$\sum_{i=1}^{r}\phi_r\sigma_r\psi^T_r$']

    Phir = Phi[:, :nr]
    Psir = PsiT[:nr, :].T
    Sigmar = np.diag(Sigma[:nr])
    ar = Psir @ Sigmar

    Phir = np.reshape(Phir[:m * n], (m, n, nr), order='F')
    Dr = np.zeros((m, n, nt, nr))
    for r in range(nr):
        aux = Phir[:, :r] @ Sigmar[:r, :r] @ Psir[:, :r].T
        Dr[:, :, :, r] = np.reshape(aux[:m * n], (m, n, nt), order='F')

    M = [Phir, ar, Dr]

    cticks = np.linspace(limits[0], limits[1], 3)
    clevels = limits
    cticksm = np.linspace(limitsm[0], limitsm[1], 3)
    clevelsm = limitsm

    fig, ax = plt.subplots(nrows, ncols, layout='tight')
    plt.tight_layout()
    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                cp0 = ax[i, j].pcolormesh(X, Y, M[j][:, :, 0, i], cmap='jet', vmin=clevelsm[0], vmax=clevelsm[1])
            elif j == 1:
                cp0 = ax[i,j].plot(t, M[j][:, i])
            elif j == 2:
                cp0 = ax[i, j].pcolormesh(X, Y, M[j][:, :, i], cmap='jet', vmin=clevels[0], vmax=clevels[1])

    def animate(it):

        for i in range(nrows):
            for j in range(ncols):
                ax[i, j].cla()

                if j == 0:
                    cp0 = ax[i, j].pcolormesh(X, Y, M[j][:, :, it, i], cmap='jet', vmin=clevelsm[0], vmax=clevelsm[1])
                    cticks_aux = cticksm
                elif j == 1:
                    cp0 = ax[i, j].plot(t[:it], M[j][:it, i], 'b-')
                elif j == 2:
                    cp0 = ax[i, j].pcolormesh(X, Y, M[j][:, :, i], cmap='jet', vmin=clevels[0], vmax=clevels[1])
                    cticks_aux = cticks

                if (j == 0) or (j == 2):
                    cp00 = ax[i, j].contourf(X, Y, B[:, :], colors='k', clevels=[1])
                    ax[i, j].axis('scaled')
                    ax[i, j].set_xlim([np.min(X), np.max(X)])
                    ax[i, j].set_ylim([np.min(Y), np.max(Y)])
                    plot_body(ax[i, j], flag_flow)

                    if i == (nrows - 1):
                        divider = make_axes_locatable(ax[i, j])
                        cax0 = divider.append_axes("bottom", size="5%", pad=0.75)
                        cbar0 = fig.colorbar(cp0, ax=ax[i, j], ticks=cticks_aux, cax=cax0, orientation="horizontal")
                        cax0.xaxis.set_ticks_position("bottom")
                else:
                    ax[i, j].set_xlim([t[0], t[-1]])
                    ax[i, j].set_ylim([-0.15, 0.15])
                    ax[i, j].set_aspect(7)

                if j == 0:
                    ax[i, j].set_ylabel('$y/D$')
                elif j == 2:
                    ax[i, j].set_yticks([])
                if i == (nrows - 1):
                    if j == 1:
                        ax[i, j].set_xlabel(r'$t/\tau$')
                    else:
                        ax[i, j].set_xlabel('$x/D$')
                else:
                    ax[i, j].set_xticks([])

                if i == 0:
                    ax[i, j].set_title(titles[j])

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate, frames=nt, interval=200)
    writergif = animation.PillowWriter(fps=2)
    anim.save(path_out, writer=writergif)