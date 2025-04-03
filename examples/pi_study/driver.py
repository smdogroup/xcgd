import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import scienceplots

# # Get ggplot colors
colors = plt.style.library["ggplot"]["axes.prop_cycle"].by_key()["color"]

plt.style.use(["science"])

if __name__ == "__main__":

    q = [1, 2, 3, 4, 5]
    n = np.logspace(np.log10(10), np.log10(3000), 30).astype(int)
    # n = np.logspace(np.log10(10), np.log10(1000), 20).astype(int)

    fig, axs = plt.subplots(ncols=2, figsize=(7.2, 3.6), constrained_layout=True)
    for index, q_ in enumerate(q):
        h_native_list = []
        h_algoim_list = []

        err_native_list = []
        err_algoim_list = []

        nquads_native_list = []
        nquads_algoim_list = []
        n_list = []

        for n_ in tqdm(n):
            cmd = ["./pi_study", "%d" % n_, "%d" % q_]
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
            h_algoim_list.append(h)

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
            lw=1.0,
            markeredgewidth=1.0,
            markersize=6.0,
            markeredgecolor="black",
            color=colors[index],
        )
        axs[1].loglog(
            np.array(h_algoim_list) ** 2,
            err_algoim_list,
            "-o",
            label="q=%d" % q_,
            alpha=0.5,
            lw=1.0,
            markeredgewidth=1.0,
            markersize=6.0,
            markeredgecolor="black",
            color=colors[index],
        )

        # axs[2].semilogx(
        #     n_list,
        #     np.array(nquads_algoim_list) / np.array(nquads_native_list),
        #     "-o",
        #     label="q=%d" % q_,
        #     alpha=0.5,
        # )

    axs[0].set_title("Gaussian Quadratures")
    axs[1].set_title("High-order Cut Quadratures")
    # axs[2].set_title("Adaptivity")

    for ax in axs[:2]:
        ax.grid()
        ax.legend()
        ax.set_xlabel(r"Mesh Size $h$")
        ax.set_ylabel(r"Numerical Integration Error $|\pi_h - \pi|$")

    # axs[2].grid()
    # axs[2].legend()
    # axs[2].set_xlabel("n")
    # axs[2].set_ylabel(r"$\dfrac{\mathrm{num~Saye's~quads}}{\mathrm{num~Gauss~quads}}$")

    fig.savefig("pi_study.pdf")
