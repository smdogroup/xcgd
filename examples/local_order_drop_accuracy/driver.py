import subprocess
import numpy as np
from matplotlib import cm
import matplotlib.patches as patches
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


def annotate_slope(
    ax, pt0, pt1, slide=0.25, scale=0.5, hoffset=0.0, voffset=-0.1, voffset_text=-0.35
):
    """
    Annotate the slope on a log-log plot

    Args:
        ax: Axes
        pt0, pt1: tuple of (x, y) where x and y are original data (not exponent)
    """

    x0, y0 = pt0
    x1, y1 = pt1

    # Make sure pt0 is always the lower one
    if y0 > y1:
        (x0, y0), (x1, y1) = (x1, y1), (x0, y0)

    dy = np.log10(y1) - np.log10(y0)
    dx = np.log10(x1) - np.log10(x0)
    slope = dy / dx

    x0 *= 10.0**hoffset
    y0 *= 10.0**voffset

    x0 = 10.0 ** (np.log10(x0) + dx * slide)
    y0 = 10.0 ** (np.log10(y0) + dy * slide)

    x1 = 10.0 ** (np.log10(x0) + dx * scale)
    y1 = 10.0 ** (np.log10(y0) + dy * scale)

    print(f"({x0:10.2e}, {y0:10.2e}), ({x1:10.2e}, {y1:10.2e}), slope: {slope:5.2f}")

    # Create a right triangle using Polygon patch
    triangle = patches.Polygon(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
        ],
        closed=True,
        # fill=False,
        edgecolor="black",
        facecolor="gray",
        zorder=100,
        lw=1,
    )

    # Add the triangle patch to the plot
    ax.add_patch(triangle)

    # Annotate the slope
    ax.annotate(
        f"{slope:.2f}",
        xy=(np.sqrt(x0 * x1), y0 * 10.0**voffset_text),
        verticalalignment="baseline",
        horizontalalignment="center",
    )

    return


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
            x = df[df["Np_bc"] == Np_bc]["h"]
            y = df[df["Np_bc"] == Np_bc]["relerr"]
            ax.loglog(
                x,
                y,
                linestyles[j],
                color=colors[i],
                clip_on=False,
                label=f"p={Np_1d - 1}, drop={Np_1d - Np_bc}",
            )

            if j == 0:
                x0, x1 = x.iloc[-2:]
                y0, y1 = y.iloc[-2:]
                annotate_slope(ax, (x0, y0), (x1, y1))

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("poisson_accuracy_study.pdf")

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
