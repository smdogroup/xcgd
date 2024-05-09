import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import mmread
from os import path
import niceplots

plt.style.use(niceplots.get_style())


def sweep(ax, Np_1d=2, exp_max=-1.0, exp_min=-5.0, num_pts=10, n=33):
    l = 1.0
    h = l / n
    r0_max = h * 2.0**0.5

    delta = []
    cond = []

    for i, d in enumerate(tqdm(np.logspace(exp_max, exp_min, num_pts))):
        r = r0_max - d * h
        prefix = "driver_output_%d" % i
        cmd = [
            "./condition_number",
            "--Np_1d=%d" % Np_1d,
            "--l=%.1f" % l,
            "--n=%d" % n,
            "--x0=%.1f" % l,
            "--y0=0.0",
            "--r=%.10f" % r,
            "--prefix=%s" % prefix,
        ]
        output = subprocess.run(cmd, capture_output=True)

        delta.append(d)
        cond.append(
            np.linalg.cond(mmread(path.join(prefix, "stiffness_matrix.mtx")).todense())
        )

    ax.loglog(
        delta, cond, "-o", label=("p=%d" % (Np_1d - 1)), clip_on=False, zorder=100
    )

    return


if __name__ == "__main__":
    fig, ax = plt.subplots()
    for Np_1d in [2, 4, 6]:
        sweep(ax=ax, Np_1d=Np_1d, exp_max=-1.0, exp_min=-5.0, num_pts=10, n=5)

    ax.set_xlabel(r"$dh$")
    ax.set_ylabel(r"condition number: $\kappa(K)$")

    ax.invert_xaxis()
    ax.legend()
    plt.show()
