import subprocess
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import json
import pandas as pd
from os.path import join
import argparse


def get_poisson_error(prefix, cmd):
    subprocess.run(cmd, capture_output=True)

    with open(join(prefix, "sol.json")) as f:
        j = json.load(f)

    return np.abs(
        np.linalg.norm(np.array(j["sol"])) / np.linalg.norm(np.array(j["sol_exact"]))
        - 1.0
    )


def run_experiments(
    Np_1d_list=[2, 4, 6, 8], max_order_drop=2, nxy_list=[8, 16, 32, 64, 128]
):
    fig, ax = plt.subplots()

    ax.set_ylabel("Normalized relative solution error")
    ax.set_xlabel("$h$")
    ax.invert_xaxis()
    ax.grid(which="both")

    order = [
        (Np_1d, Np_bc)
        for Np_1d in Np_1d_list
        for Np_bc in reversed(range(max(2, Np_1d - max_order_drop), Np_1d + 1))
    ]

    df_data = {
        "Np_1d": [],
        "Np_bc": [],
        "nxy": [],
        "h": [],
        "relerr": [],
    }

    for Np_1d, Np_bc in order:
        for nxy in nxy_list:
            print(f"Np_1d: {Np_1d:2d}, Np_bc: {Np_bc:2d}, nxy: {nxy:4d}")

            prefix = f"outputs_Np1d_{Np_1d}_Npbc_{Np_bc}_nxy_{nxy}"
            cmd = [
                "./poisson_order_drop",
                "--save-degenerate-stencils=0",
                f"--Np_1d={Np_1d}",
                f"--Np_bc={Np_bc}",
                f"--nxy={nxy}",
                f"--prefix={prefix}",
            ]

            err = get_poisson_error(prefix, cmd)

            df_data["Np_1d"].append(Np_1d)
            df_data["Np_bc"].append(Np_bc)
            df_data["nxy"].append(nxy)
            df_data["h"].append(2.0 / nxy)
            df_data["relerr"].append(err)

    return pd.DataFrame(df_data)


def plot(cases_df):
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(6.4, 4.8),
        constrained_layout=True,
    )

    Np_1d_list = sorted(list(set(cases_df["Np_1d"])))
    colors = cm.tab10(range(10))
    linestyles = [i + j for j in ["o", "s", "<", ">"] for i in ["-"]]

    # Cases for each Np
    for i, Np_1d in enumerate(Np_1d_list):
        df = cases_df[cases_df["Np_1d"] == Np_1d]

        Np_bc_list = sorted(list(set(df["Np_bc"])), reverse=True)

        for j, Np_bc in enumerate(Np_bc_list):
            ax.loglog(
                df[df["Np_bc"] == Np_bc]["h"],
                df[df["Np_bc"] == Np_bc]["relerr"],
                linestyles[j],
                color=colors[i],
                clip_on=False,
                label=f"p={Np_1d - 1}, drop={Np_1d - Np_bc}",
            )

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()
    plt.savefig("accuracy_study.pdf")

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.HelpFormatter)

    p.add_argument("--csv", type=str, default=None, help="case data csv")
    p.add_argument(
        "--Np_1d",
        nargs="+",
        type=int,
        default=[2, 4, 6],
        help="list of Np_1d to use",
    )
    p.add_argument(
        "--max-order-drop",
        type=int,
        default=2,
        help="maximum order to drop for boundary elements",
    )
    p.add_argument(
        "--nxy",
        nargs="+",
        default=[8, 16, 32, 64, 128, 256],
        type=int,
        help="list of number of mesh elements per dimension",
    )

    args = p.parse_args()

    if args.csv is None:
        df = run_experiments(
            Np_1d_list=args.Np_1d,
            max_order_drop=args.max_order_drop,
            nxy_list=args.nxy,
        )
        df.to_csv("cases_data.csv")
    else:
        df = pd.read_csv(args.csv)

    plot(df)
