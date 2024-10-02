import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas
from os.path import join


if __name__ == "__main__":
    fig, ax = plt.subplots()

    p_list = [1, 3, 5]

    for nxy in [8, 16, 32]:
        err_l2_list = []
        for p in tqdm(p_list):
            Np_1d = p + 1
            prefix = f"outputs_Np_{Np_1d}_nxy_{nxy}"

            cmd = [
                "./poisson_order_drop",
                f"--Np_1d={Np_1d}",
                f"--nxy={nxy}",
                f"--prefix={prefix}",
            ]

            subprocess.run(cmd, capture_output=True)

            with open(join(prefix, "sol.json")) as f:
                j = json.load(f)

            err_l2_list.append(
                np.abs(
                    np.linalg.norm(np.array(j["sol"]))
                    / np.linalg.norm(np.array(j["sol_exact"]))
                    - 1.0
                )
            )

        ax.semilogy(p_list, err_l2_list, "-o", clip_on=False, label=f"nxy={nxy}")

    ax.set_ylabel("Relative solution error")
    ax.set_xlabel("$p$")

    ax.grid()
    ax.legend()
    plt.savefig("accuracy_study.pdf")
