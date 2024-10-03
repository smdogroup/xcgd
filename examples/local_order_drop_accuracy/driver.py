import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd
from os.path import join


def get_poisson_error(prefix, cmd):
    subprocess.run(cmd, capture_output=True)

    with open(join(prefix, "sol.json")) as f:
        j = json.load(f)
    return np.abs(
        np.linalg.norm(np.array(j["sol"])) / np.linalg.norm(np.array(j["sol_exact"]))
        - 1.0
    )


def run_experiments():
    fig, ax = plt.subplots()

    ax.set_ylabel("Normalized relative solution error")
    ax.set_xlabel("$h$")
    ax.invert_xaxis()
    ax.grid(which="both")

    order = [
        (Np_1d, Np_bc)
        for Np_1d in [2, 4, 6, 8]
        for Np_bc in reversed(range(max(2, Np_1d - 2), Np_1d + 1))
    ]

    for Np_1d, Np_bc in order:
        nxy_list = [8, 16, 32, 64, 128]
        h_list = [2.0 / nxy for nxy in nxy_list]

        # polulate error for a single sweep
        normalized_err_list = []
        bl_err = None
        for nxy in nxy_list:
            print(f"Np_1d: {Np_1d:2d}, Np_bc: {Np_bc:2d}, nxy: {nxy:4d}")

            prefix = f"outputs_Np1d_{Np_1d}_Npbc_{Np_bc}_nxy_{nxy}"
            cmd = [
                "./poisson_order_drop",
                f"--Np_1d={Np_1d}",
                f"--Np_bc={Np_bc}",
                f"--nxy={nxy}",
                f"--prefix={prefix}",
            ]

            err = get_poisson_error(prefix, cmd)

            if bl_err is None:
                bl_err = err
            normalized_err_list.append(err / bl_err)

        ax.loglog(
            h_list,
            normalized_err_list,
            "-o",
            clip_on=False,
            label=f"p={Np_1d - 1}, drop={Np_1d - Np_bc}",
        )
        ax.legend()
        plt.savefig("accuracy_study.pdf")

    return


if __name__ == "__main__":
    run_experiments()
