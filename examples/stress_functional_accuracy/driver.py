import numpy as np
import pandas as pd
import argparse
from time import time
import subprocess
from os.path import join
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from typing import List


def save_cmd(prefix, cmd: List[str]):
    with open(os.path.join(prefix, "cmd.txt"), "w") as f:
        f.write(" ".join(cmd))
        f.write("\n")


def print_and_log(logpath, string):
    with open(logpath, "a") as f:
        f.write(string)
        f.write("\n")
    print(string)


def execute(prefix, cmd):
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

    return j


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
    mesh,
    physics,
    instance,
    save_vtk: bool,
    use_ersatz: bool,
    ersatz_ratio: float,
    nitsche_eta: float,
    smoke: bool,
):
    logpath = os.path.join(run_name, f"{run_name}.log")
    open(logpath, "w").close()  # erase existing file

    df_data = {
        "Np_1d": [],
        "h": [],
        "stress_norm": [],
    }

    if physics == "poisson":
        df_data["val_norm"] = []
        df_data["energy_norm"] = []

    Np_1d_list = [2, 4, 6]
    nxy_list = list(map(int, np.logspace(3, 7, 20, base=2)))
    if smoke:
        Np_1d_list = [2, 4]
        nxy_list = [4, 8, 16, 32]

    for Np_1d in Np_1d_list:
        for nxy in nxy_list:
            prefix = os.path.join(run_name, f"Np_{Np_1d}_nxy_{nxy}")
            if not os.path.isdir(prefix):
                os.mkdir(prefix)
            cmd = [
                "./stress_functional_accuracy",
                f"--physics={physics}",
                f"--instance={instance}",
                f"--use-finite-cell-mesh={1 if mesh == 'finite-cell-mesh' else 0}",
                f"--Np_1d={Np_1d}",
                f"--nxy={nxy}",
                f"--prefix={prefix}",
                f"--use-ersatz={int(use_ersatz)}",
                f"--ersatz-ratio={ersatz_ratio}",
                f"--nitsche-eta={nitsche_eta}",
                f"--save-vtk={int(save_vtk)}",
            ]

            save_cmd(prefix, cmd)

            t1 = time()
            j = execute(prefix, cmd)
            t2 = time()

            print_and_log(
                logpath,
                f"Np_1d: {Np_1d:2d}, nxy: {nxy:4d}, execution time: {t2 - t1:.2f} s",
            )

            df_data["Np_1d"].append(Np_1d)
            df_data["h"].append(1.0 / nxy)
            df_data["stress_norm"].append(j["stress_norm"])

            if physics == "poisson":
                df_data["val_norm"].append(j["val_norm"])
                df_data["energy_norm"].append(j["energy_norm"])

    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(run_name, f"{run_name}.csv"), index=False)
    return df


def plot_poisson(df, voffset, voffset_text):
    fig, axs = plt.subplots(
        ncols=3,
        nrows=1,
        figsize=(19.2, 4.8),
        constrained_layout=True,
    )

    for Np_1d, sub_df in df.groupby("Np_1d"):
        for key, ax in zip(["val_norm", "stress_norm", "energy_norm"], axs):
            # Get averaged slope
            x = sub_df["h"]
            y = sub_df[key]
            slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)
            label = f"$p={Np_1d - 1}, \Delta:{slope:.2f}$"
            ax.loglog(x, y, "-o", label=label)
            x0, x1 = x.iloc[-2:]
            y0, y1 = y.iloc[-2:]
            annotate_slope(
                ax, (x0, y0), (x1, y1), voffset=voffset, voffset_text=voffset_text
            )

    for ylabel, ax in zip(
        [
            r"$\left[\int_h (u - u_h)^2 d\Omega\right]^{1/2}$",
            r"$\left[\int_h (\mathbf{\sigma} - \mathbf{\sigma}_h)^2 d\Omega\right]^{1/2}$",
            r"$\left[\int_h ((u - u_h)^2 + (\mathbf{\sigma} - \mathbf{\sigma}_h)^2) d\Omega\right]^{1/2}$",
        ],
        axs,
    ):
        ax.grid(which="both")
        ax.legend()
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(ylabel)

    return fig, axs


def plot_elasticity(df, voffset, voffset_text):
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(6.4, 4.8),
        constrained_layout=True,
    )

    for Np_1d, sub_df in df.groupby("Np_1d"):
        # Get averaged slope
        x = sub_df["h"]
        y = sub_df["stress_norm"]
        slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)
        label = f"$p={Np_1d - 1}, \Delta:{slope:.2f}$"
        ax.loglog(x, y, "-o", label=label)
        x0, x1 = x.iloc[-2:]
        y0, y1 = y.iloc[-2:]
        annotate_slope(
            ax, (x0, y0), (x1, y1), voffset=voffset, voffset_text=voffset_text
        )

        ylabel = r"$\left[\int_h  \text{tr}((\mathbf{S} - \mathbf{S}_h)^T(\mathbf{S} - \mathbf{S}_h)) d\Omega\right]^{1/2}$"

        ax.grid(which="both")
        ax.legend()
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(ylabel)

    return fig, ax


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--physics",
        default="poisson",
        choices=[
            "poisson",
            "elasticity-mms",
            "elasticity-bulk",
            "elasticity-interface",
        ],
    )
    p.add_argument("--instance", default="square", choices=["square", "circle"])
    p.add_argument(
        "--mesh", default="cut-mesh", choices=["cut-mesh", "finite-cell-mesh"]
    )
    p.add_argument("--csv", type=str)
    p.add_argument("--voffset", default=-0.2, type=float)
    p.add_argument("--voffset_text", default=-0.4, type=float)
    p.add_argument("--save-vtk", action="store_true")
    p.add_argument("--use-ersatz", action="store_true")
    p.add_argument("--ersatz-ratio", default=1e-6, type=float)
    p.add_argument("--nitsche-eta", default=1e8, type=float)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    # Sanity checks
    if args.physics == "elasticity-mms" and args.instance == "square":
        print(f"[Error] square is not implemented for elasticity-mms")
        exit(-1)
    if args.physics != "poisson" and args.instance != "square":
        print(
            f"[Warning] instance {args.instance} has no effect for --physics {args.physics}"
        )
    if args.physics != "elasticity-bulk" and args.use_ersatz:
        print(f"[Warning] physics {args.physics} does not have --use_ersatz option")
    if args.instance == "square" and args.mesh == "finite-cell-mesh":
        print(
            f"[Warning] option --mesh does not have effect for {args.instance} instance"
        )

    if args.physics == "poisson" or args.physics == "elasticity-mms":
        run_name = f"{args.physics}_energy_precision_{args.mesh}_{args.instance}"
        if args.instance == "circle":
            run_name += f"_nitscheeta_{args.nitsche_eta:.0e}"

    else:
        run_name = f"{args.physics}_energy_precision_{args.mesh}"

    if args.physics == "elasticity" and args.use_ersatz:
        run_name += f"_ersatz_{args.ersatz_ratio}"

    if args.smoke_test:
        run_name = "smoke_" + run_name

    if not os.path.isdir(run_name):
        os.mkdir(run_name)

    if args.csv is None:
        df = run_experiments(
            run_name,
            args.mesh,
            args.physics,
            args.instance,
            args.save_vtk,
            args.use_ersatz,
            args.ersatz_ratio,
            args.nitsche_eta,
            args.smoke_test,
        )
    else:
        df = pd.read_csv(args.csv)
    print(df)

    if args.physics == "poisson":
        fig, _ = plot_poisson(df, args.voffset, args.voffset_text)
    else:
        fig, _ = plot_elasticity(df, args.voffset, args.voffset_text)

    fig.savefig(os.path.join(run_name, f"{run_name}.pdf"))
