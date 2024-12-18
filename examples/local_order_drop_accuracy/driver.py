import subprocess
from niceplots.utils import adjust_spines
import numpy as np
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json
import pandas as pd
from os.path import join
import argparse
from time import time
import sys


def print_and_log(logpath, string):
    with open(logpath, "a") as f:
        f.write(string)
        f.write("\n")
    print(string)


def get_l2_error(prefix, cmd):
    try:
        subprocess.run(cmd, capture_output=True)
    # except subprocess.CalledProcessError as e:
    except Exception as e:
        print("execution of the following command has failed:")
        print(" ".join(cmd))
        print("Below is the error info")
        print(e)

    with open(join(prefix, "sol.json")) as f:
        j = json.load(f)

    err_l2norm = j["err_l2norm"]
    err_l2norm_nrmed = j["err_l2norm"] / j["l2norm"]
    return err_l2norm, err_l2norm_nrmed


def annotate_slope(
    ax, pt0, pt1, slide=0.15, scale=0.7, hoffset=0.0, voffset=-0.1, voffset_text=-0.35
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
    run_name,
    physics_type="poisson",
    Np_1d_list=[2, 4, 6, 8],
    max_order_drop=2,
    nxy_list=[8, 16, 32, 64, 128],
):
    logpath = f"{run_name}.log"
    open(logpath, "w").close()  # erase existing file

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
        "err_l2norm": [],
        "err_l2norm_nrmed": [],
    }

    for Np_1d, Np_bc in order:
        for nxy in nxy_list:
            prefix = f"outputs_{physics_type}_Np1d_{Np_1d}_Npbc_{Np_bc}_nxy_{nxy}"
            cmd = [
                "./order_drop_study",
                f"--physics_type={physics_type}",
                "--save-degenerate-stencils=0",
                f"--Np_1d={Np_1d}",
                f"--Np_bc={Np_bc}",
                f"--nxy={nxy}",
                f"--prefix={prefix}",
            ]

            t1 = time()
            err_l2norm, err_l2norm_nrmed = get_l2_error(prefix, cmd)
            t2 = time()

            print_and_log(
                logpath,
                f"Np_1d: {Np_1d:2d}, Np_bc: {Np_bc:2d}, nxy: {nxy:4d}, execution time: {t2 - t1:.2f} s",
            )

            df_data["Np_1d"].append(Np_1d)
            df_data["Np_bc"].append(Np_bc)
            df_data["nxy"].append(nxy)
            df_data["h"].append(2.0 / nxy)
            df_data["err_l2norm"].append(err_l2norm)
            df_data["err_l2norm_nrmed"].append(err_l2norm_nrmed)

    return pd.DataFrame(df_data)


def adjust_plot_lim(ax, left=0.0, right=0.2, bottom=0.3, up=0.0):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    dx = np.log10(xmax) - np.log10(xmin)
    dy = np.log10(ymax) - np.log10(ymin)

    xmin /= 10.0 ** (dx * left)
    xmax *= 10.0 ** (dx * right)

    ymin /= 10.0 ** (dy * bottom)
    ymax *= 10.0 ** (dy * up)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    return


def plot(cases_df, physics_type, normalize, voffset, voffset_text):
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
            # x = df[df["Np_bc"] == Np_bc]["nxy"]
            if normalize:
                y = df[df["Np_bc"] == Np_bc]["err_l2norm_nrmed"]
            else:
                y = df[df["Np_bc"] == Np_bc]["err_l2norm"]

            # Get averaged slope
            slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)

            label = f"$p={Np_1d - 1}, p_{{bc}}={Np_bc-1}, \Delta:{slope:.2f}$"
            ax.loglog(x, y, linestyles[j], color=colors[i], clip_on=False, label=label)

            print(label)

            if j == 0:
                x0, x1 = x.iloc[-2:]
                y0, y1 = y.iloc[-2:]
                annotate_slope(
                    ax, (x0, y0), (x1, y1), voffset=voffset, voffset_text=voffset_text
                )

    ax.set_xlabel("Mesh size $h$")
    # ax.set_xlabel("number of elements in each dimension")

    if physics_type == "poisson":
        numerator = r"\sqrt{\int_\Omega (u_h - u_\text{exact})^2 d\Omega}"
        denominator = r"\sqrt{\int_\Omega u_\text{exact}^2 d\Omega}"
        if normalize:
            ax.set_ylabel(
                "Normalized L2 norm of the solution error\n"
                + r"$\dfrac{"
                + numerator
                + r"}{"
                + denominator
                + r"}$"
            )
        else:
            ax.set_ylabel("L2 norm of the solution error\n" + r"$" + numerator + r"$")
    else:
        numerator = r"\sqrt{\int_\Omega (\mathbf{u}_h - \mathbf{u}_\text{exact}) \cdot (\mathbf{u}_h - \mathbf{u}_\text{exact}) d\Omega}"
        denominator = r"\sqrt{\int_\Omega \mathbf{u}_\text{exact} \cdot \mathbf{u}_\text{exact} d\Omega}"
        if normalize:
            ax.set_ylabel(
                "Normalized L2 norm of the solution error\n"
                + r"$\dfrac{"
                + numerator
                + r"}{"
                + denominator
                + r"}$"
            )
        else:
            ax.set_ylabel("L2 norm of the solution error\n" + r"$" + numerator + r"$")

    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    adjust_plot_lim(ax, left=0.0, right=0.0, bottom=0.0, up=0.0)

    return fig, ax


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.HelpFormatter)

    p.add_argument(
        "--physics_type", default="poisson", choices=["poisson", "linear_elasticity"]
    )
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
        default=[8, 16, 32, 64, 128],
        type=int,
        help="list of number of mesh elements per dimension",
    )
    p.add_argument("--normalize-l2error", action="store_true")
    p.add_argument("--voffset", default=-0.1, type=float)
    p.add_argument("--voffset_text", default=-0.1, type=float)

    args = p.parse_args()

    run_name = f"precision_order_drop_{args.physics_type}"
    if args.normalize_l2error:
        run_name += "_nrmed"

    if args.csv is None:
        df = run_experiments(
            run_name,
            args.physics_type,
            Np_1d_list=args.Np_1d,
            max_order_drop=args.max_order_drop,
            nxy_list=args.nxy,
        )
    else:
        df = pd.read_csv(args.csv)
        drop_column = "Unnamed: 0"
        if drop_column in df.columns:
            df = df.drop(columns=[drop_column])
        df = df.sort_values(
            ["Np_1d", "Np_bc", "nxy"], ascending=[True, False, True]
        ).reset_index(drop=True)

        # with pd.option_context("display.max_rows", None, "display.max_columns", None):
        #     print(df)
        #     exit()

    fig, ax = plot(
        df, args.physics_type, args.normalize_l2error, args.voffset, args.voffset_text
    )

    fig.savefig(f"{run_name}.pdf")
    df.to_csv(f"{run_name}.csv", index=False)

    with open(f"{run_name}.txt", "w") as f:
        f.write("python " + " ".join(sys.argv))
