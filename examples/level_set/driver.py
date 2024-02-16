import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":

    q = [1, 2, 3, 4, 5]
    n = np.logspace(np.log10(10), np.log10(3000), 30).astype(int)
    # n = np.logspace(np.log10(10), np.log10(1000), 20).astype(int)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True)
    for q_ in q:
        h_native_list = []
        h_algoim_list = []

        err_native_list = []
        err_algoim_list = []

        nquads_native_list = []
        nquads_algoim_list = []
        n_list = []

        for n_ in tqdm(n):
            cmd = ["./level_set", "%d" % n_, "%d" % q_]
            output = subprocess.run(cmd, capture_output=True)

            h, err_native, nquads_native, err_algoim, nquads_algoim = (
                output.stdout.decode().strip().split(",")
            )

            h = float(h)
            err_native = float(err_native)
            nquads_native = int(nquads_native)
            err_algoim = float(err_algoim)
            nquads_algoim = int(nquads_algoim)

            h_native_list.append(h)
            h_algoim_list.append(h * np.sqrt(nquads_native / nquads_algoim))

            err_native_list.append(err_native)
            err_algoim_list.append(err_algoim)

            nquads_native_list.append(nquads_native)
            nquads_algoim_list.append(nquads_algoim)
            n_list.append(n_)

        axs[0].loglog(
            np.array(h_native_list) ** 2,
            err_native_list,
            "-o",
            label="q=%d" % q_,
            alpha=0.5,
        )
        axs[1].loglog(
            np.array(h_algoim_list) ** 2,
            err_algoim_list,
            "-o",
            label="q=%d" % q_,
            alpha=0.5,
        )

        axs[2].semilogx(
            n_list,
            np.array(nquads_algoim_list) / np.array(nquads_native_list),
            "-o",
            label="q=%d" % q_,
            alpha=0.5,
        )

    axs[0].set_title("Gaussian quadratures")
    axs[1].set_title("Saye's quadratures")
    axs[2].set_title("Adaptivity")

    for ax in axs[:2]:
        ax.grid()
        ax.legend()
        ax.set_xlabel(r"$h^2$")
        ax.set_ylabel(r"$|\pi - \pi_{exact}|$")

    axs[2].grid()
    axs[2].legend()
    axs[2].set_xlabel("n")
    axs[2].set_ylabel(r"$\dfrac{\mathrm{num~Gauss~quads}}{\mathrm{num~Saye's~quads}}$")

    fig.savefig("pi_study.pdf")
    plt.show()
