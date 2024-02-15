import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_pi(xmin=-0.6, xmax=0.6, nelems_x=10, deg=2):
    pi = 0.0
    h = (xmax - xmin) / nelems_x
    detJ = h**2 / 4.0

    X = np.linspace(xmin, xmax, nelems_x + 1)

    qpts, wts = np.polynomial.legendre.leggauss(deg)
    Xqpts = np.tile(X[:-1, np.newaxis], (1, len(qpts)))
    wts = np.tile(wts, (nelems_x, 1))
    Xqpts += (qpts + 1.0) * h / 2.0

    Xq, Yq = np.meshgrid(Xqpts.flatten(), Xqpts.flatten())
    wts = np.outer(wts.flatten(), wts.flatten())

    pi = detJ * np.sum((Xq**2 + Yq**2 < 1.0) * wts)

    return h, pi


if __name__ == "__main__":
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    xmax = 1.0
    xmin = -1.0

    p = [2, 3, 4, 5, 6, 7, 8]
    for p_ in p:
        h = []
        err = []
        for n in tqdm([10, 50, 100, 500, 1000, 2000, 3000]):
            h_, pi = compute_pi(xmin=xmin, xmax=xmax, nelems_x=n, deg=p_)
            err.append(np.abs(np.pi - pi))
            h.append(h_)
        axs[0].loglog(h, err, "-o", label="p=%d" % p_)

    axs[0].set_xlabel("h (nelems = (1/h)^2)")
    axs[0].set_ylabel("error")

    n = [10, 50, 100, 500, 1000]
    p = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for n_ in n:
        err = []
        for p_ in tqdm(p):
            _, pi = compute_pi(xmin=xmin, xmax=xmax, nelems_x=n_, deg=p_)
            err.append(np.abs(np.pi - pi))

        axs[1].semilogy(p, err, "-o", label="h={:.1e}".format((xmax - xmin) / n_))
        axs[1].set_xlabel("p")
        axs[1].set_ylabel("error")

    for ax in axs:
        ax.grid()
        ax.legend()

    plt.show()
