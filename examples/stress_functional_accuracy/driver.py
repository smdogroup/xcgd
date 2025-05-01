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

import scienceplots

# # Get ggplot colors
colors = plt.style.library["ggplot"]["axes.prop_cycle"].by_key()["color"]

plt.style.use(["science"])

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
    ax, pt0, pt1, slide=0.05, scale=0.9, hoffset=0.0, voffset=-0.1, voffset_text=-0.35
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
        lw=0.5,
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


def annotate_averaged_slope(ax, x, y, voffset):
    """
    annotate the averaged slope on a log-log plot given a list of x and y values.
    """
    slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)

    xc = np.sqrt(np.min(x) * np.max(x))  # find center
    yc = np.sqrt(np.min(y) * np.max(y))

    delta = 0.5 * np.log10(np.max(x) / np.min(x))

    # Find voffset
    xi = np.log10(x)
    eta = np.log10(y)

    xic = np.log10(xc)
    etac = np.log10(yc)

    voffset += np.min(eta - slope * (xi - xic) - etac)
    yc *= 10.0**voffset

    t = len(x) - 1.0
    FRAC_LEFT = (t / 2.0 - 0.5) / (t / 2.0)
    FRAC_RIGHT = -(t / 2.0 - 2.5) / (t / 2.0)

    x0 = xc / 10.0 ** (delta * FRAC_LEFT)
    x1 = xc * 10.0 ** (delta * FRAC_RIGHT)
    y0 = yc / 10.0 ** (delta * FRAC_LEFT * slope)
    y1 = yc * 10.0 ** (delta * FRAC_RIGHT * slope)

    # Make sure pt0 is always the lower one
    if y0 > y1:
        (x0, y0), (x1, y1) = (x1, y1), (x0, y0)

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
        facecolor="#d4d4d4",
        zorder=100,
        lw=0.5,
    )

    # Add the triangle patch to the plot
    ax.add_patch(triangle)

    # Annotate the slope
    ax.annotate(
        f"{slope:.2f}",
        # xy=(np.sqrt(x0 * x1), y0 * 10.0**voffset_text),
        xy=(np.sqrt(x0 * x1), y0),
        xytext=(0, -2),  # -2 points down
        textcoords="offset points",  # interpret xytext as offset in points
        verticalalignment="top",
        horizontalalignment="center",
    )
    return


def expand_logy_bottom(ax, frac=0.05):
    """
    Expand bottom of a plot by given percent, where y is in log scale
    """
    ymin, ymax = ax.get_ylim()
    t = np.log10(ymax / ymin)
    dt = frac * t
    ax.set_ylim(bottom=ymin * 10.0 ** (-dt))

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
    Np_1d_list: List[int],
    nxy_min: int,
    nxy_max: int,
    nxy_num: int,
):
    logpath = os.path.join(run_name, f"{run_name}.log")
    open(logpath, "w").close()  # erase existing file

    df_data = {
        "Np_1d": [],
        "h": [],
        "total_time": [],
        "sol_time": [],
        "jacobian_time": [],
        "residual_time": [],
        "chol_init_time": [],
        "chol_factor_time": [],
        "chol_solve_time": [],
    }

    if physics == "poisson":
        df_data["val_norm"] = []
        df_data["energy_norm"] = []

    elif physics == "elasticity-interface":
        df_data["stress_norm_primary"] = []
        df_data["stress_norm_secondary"] = []
        df_data["stress_norm_primary_interface"] = []
        df_data["stress_norm_secondary_interface"] = []
    else:
        df_data["stress_norm"] = []

    nxy_list = list(
        map(round, np.logspace(np.log2(nxy_min), np.log2(nxy_max), nxy_num, base=2))
    )

    print(f"sweeping nxy_list: {nxy_list}")

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
            df_data["total_time"].append(j["total_time"])
            df_data["sol_time"].append(j["sol_time"])
            df_data["jacobian_time"].append(j["jacobian_time"])
            df_data["residual_time"].append(j["residual_time"])
            df_data["chol_init_time"].append(j["chol_init_time"])
            df_data["chol_factor_time"].append(j["chol_factor_time"])
            df_data["chol_solve_time"].append(j["chol_solve_time"])

            if physics == "poisson":
                df_data["val_norm"].append(j["val_norm"])
                df_data["energy_norm"].append(j["energy_norm"])
            elif physics == "elasticity-interface":
                df_data["stress_norm_primary"].append(j["stress_norm_primary"])
                df_data["stress_norm_secondary"].append(j["stress_norm_secondary"])
                df_data["stress_norm_primary_interface"].append(
                    j["stress_norm_primary_interface"]
                )
                df_data["stress_norm_secondary_interface"].append(
                    j["stress_norm_secondary_interface"]
                )
            else:
                df_data["stress_norm"].append(j["stress_norm"])

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
            label = f"$p={Np_1d - 1}$"
            ax.loglog(x, y, "-o", label=label)
            annotate_averaged_slope(ax, x, y, voffset)

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


def plot_elasticity_interface(df, what, voffset, voffset_text):
    assert what != "cpu_time" and what != "cpu_time_breakdown"

    fig, axs = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(10.6, 8.0),
        constrained_layout=True,
    )

    if what == "stress_time":
        xlabel = r"CPU time (s)"
    else:
        xlabel = r"$h$"

    if what == "roi":
        ylabels = [
            r"ROI: $\dfrac{1}{\text{CPU time} \cdot \text{primary error norm}}$",
            r"ROI: $\dfrac{1}{\text{CPU time} \cdot \text{secondary error norm}}$",
            r"ROI: $\dfrac{1}{\text{CPU time} \cdot \text{primary interface error norm}}$",
            r"ROI: $\dfrac{1}{\text{CPU time} \cdot \text{secondary interface error norm}}$",
        ]
    else:
        ylabels = [
            r"$\left[\int_{\text{primary mesh},h}  \text{tr}((\mathbf{S} - \mathbf{S}_h)^T(\mathbf{S} - \mathbf{S}_h)) d\Omega\right]^{1/2}$",
            r"$\left[\int_{\text{secondary mesh},h}  \text{tr}((\mathbf{S} - \mathbf{S}_h)^T(\mathbf{S} - \mathbf{S}_h)) d\Omega\right]^{1/2}$",
            r"$\left[\int_{\text{primary mesh}, h}  \text{tr}((\mathbf{S} - \mathbf{S}_h)^T(\mathbf{S} - \mathbf{S}_h)) d\Gamma\right]^{1/2}$",
            r"$\left[\int_{\text{secondary mesh}h}  \text{tr}((\mathbf{S} - \mathbf{S}_h)^T(\mathbf{S} - \mathbf{S}_h)) d\Gamma\right]^{1/2}$",
        ]

    axs = axs.flatten()

    for i, (Np_1d, sub_df) in enumerate(df.groupby("Np_1d")):
        for key, ax in zip(
            [
                "stress_norm_primary",
                "stress_norm_secondary",
                "stress_norm_primary_interface",
                "stress_norm_secondary_interface",
            ],
            axs,
        ):
            if what == "stress_time":
                x = sub_df["total_time"]
            else:
                x = sub_df["h"]

            if what == "roi":
                y = 1.0 / sub_df[key] / sub_df["total_time"]
            else:
                y = sub_df[key]

            # Get averaged slope
            slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)
            label = f"$p={Np_1d - 1}$"

            ax.loglog(
                x,
                y,
                "-o",
                label=label,
                lw=1.0,
                markeredgewidth=1.0,
                markersize=6.0,
                markeredgecolor="black",
                color=colors[i],
            )

    for ylabel, title, key, ax in zip(
        ylabels,
        [
            "Stress Error On the Primary Mesh",
            "Stress Error On the Secondary Mesh",
            "Stress Error On Interface from the Primary Mesh",
            "Stress Error On Interface from the Secondary Mesh",
        ],
        [
            "stress_norm_primary",
            "stress_norm_secondary",
            "stress_norm_primary_interface",
            "stress_norm_secondary_interface",
        ],
        axs,
    ):

        ymin, ymax = ax.get_ylim()
        v_off = -np.log10(ymax / ymin) * 0.02
        v_off_txt = -np.log10(ymax / ymin) * 0.035
        ax.set_ylim(bottom=ymin * 10.0 ** (v_off_txt * 1.05))

        for Np_1d, sub_df in df.groupby("Np_1d"):
            if what == "stress_time":
                x = sub_df["total_time"]
            else:
                x = sub_df["h"]

            if what == "roi":
                y = 1.0 / sub_df[key] / sub_df["total_time"]
            else:
                y = sub_df[key]

            annotate_averaged_slope(ax, x, y, v_off * voffset)

        if what != "stress_time":
            # Remove all existing ticks
            ax.tick_params(axis="x", which="both", length=0, labelbottom=False)

            # Set new ticks with explicit positions and labels
            ax.set_xticks(df["h"].drop_duplicates())
            ax.set_xticklabels(
                df["h"].drop_duplicates().apply(lambda x: f"{x:.1e}"),
                rotation=45,
                ha="right",
            )

            # Add ticks
            ax.tick_params(
                axis="x", which="major", direction="in", length=3, labelbottom=True
            )

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    return fig, axs


def plot_cpu_time(df, voffset, voffset_text):
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(5.3, 4.0),
        constrained_layout=True,
    )
    xlabel = r"$h$"
    ylabel = r"CPU time (s)"

    for i, (Np_1d, sub_df) in enumerate(df.groupby("Np_1d")):
        x = sub_df["h"]
        y = sub_df["total_time"]
        slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)
        label = f"$p={Np_1d - 1}$"
        ax.loglog(
            x,
            y,
            "-o",
            label=label,
            lw=1.0,
            markeredgewidth=1.0,
            markersize=6.0,
            markeredgecolor="black",
            color=colors[i],
        )

    ymin, ymax = ax.get_ylim()
    v_off = -np.log10(ymax / ymin) * 0.03
    expand_logy_bottom(ax, frac=voffset_text * 0.07)

    # Annotate the slopes
    for Np_1d, sub_df in df.groupby("Np_1d"):
        x = sub_df["h"]
        y = sub_df["total_time"]
        annotate_averaged_slope(ax, x, y, v_off * voffset)

    # Remove all existing ticks
    ax.tick_params(axis="x", which="both", length=0, labelbottom=False)

    # Set new ticks with explicit positions and labels
    ax.set_xticks(df["h"].drop_duplicates())
    ax.set_xticklabels(
        df["h"].drop_duplicates().apply(lambda x: f"{x:.1e}"),
        rotation=45,
        ha="right",
    )

    # Add ticks
    ax.tick_params(axis="x", which="major", direction="in", length=3, labelbottom=True)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_cpu_time_breakdown(df):
    num_Np_1d = len(set(df["Np_1d"]))
    fig, axs = plt.subplots(
        ncols=num_Np_1d,
        nrows=1,
        figsize=(5.3 * num_Np_1d, 4.0),
        constrained_layout=True,
    )

    yname_label_map = {
        "jacobian_time": "Jacobian assembly",
        "residual_time": "Residual assembly",
        "chol_init_time": "Cholesky initialization",
        "chol_factor_time": "Cholesky factorization",
        "chol_solve_time": "Cholesky solve",
        "other": "Other",
    }

    # Derived column
    df["other"] = df["sol_time"]
    for yname in yname_label_map.keys():
        if yname != "other":
            df["other"] -= df[yname]

    for i, (Np_1d, sub_df) in enumerate(df.groupby("Np_1d")):

        # Determine the bar width
        bar_width = (
            0.8
            * (np.log10(max(sub_df["h"])) - np.log10(min(sub_df["h"])))
            / len(sub_df["h"])
            / len(yname_label_map)
        ) * np.log(10)

        for j, (yname, label) in enumerate(yname_label_map.items()):
            axs[i].bar(
                sub_df["h"]
                + (j - (len(yname_label_map) - 1) / 2) * bar_width * sub_df["h"],
                sub_df[yname],
                bar_width
                * sub_df["h"],  # transformation from linear space to log space
                label=label if i == 0 else None,
                facecolor=colors[j],
                edgecolor="black",
                linewidth=0.5,
                alpha=1.0,
            )

        axs[i].set_xscale("log")
        axs[i].set_yscale("log")

        # Remove all existing ticks
        axs[i].tick_params(axis="x", which="both", length=0, labelbottom=False)

        # Set new ticks with explicit positions and labels
        axs[i].set_xticks(sub_df["h"])
        axs[i].set_xticklabels(
            sub_df["h"].apply(lambda x: f"{x:.1e}"), rotation=45, ha="right"
        )

        # Make ticks point outward/downward
        axs[i].tick_params(
            axis="x", which="major", direction="out", length=3, labelbottom=True
        )

        # Remove the ticks from top, if any
        axs[i].tick_params(top=False)

        axs[i].set_xlabel(r"$h$")
        axs[i].set_ylabel(r"CPU time (s)")
        axs[i].set_title(f"$p={Np_1d - 1}$")

    fig.legend(
        loc="upper center",
        ncols=6,
        bbox_to_anchor=(0.5, 1.07),
        bbox_transform=fig.transFigure,
    )

    return fig, axs


def plot_elasticity(df, what, voffset, voffset_text):
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(5.3, 4.0),
        constrained_layout=True,
    )
    assert what != "cpu_time" and what != "cpu_time_breakdown"

    if what == "stress_time":
        xlabel = r"CPU time (s)"
    else:
        xlabel = r"$h$"

    if what == "roi":
        ylabel = r"ROI: $\dfrac{1}{\text{CPU time} \cdot \text{error norm}}$"
    else:
        ylabel = r"$\left[\int_h  \text{tr}((\mathbf{S} - \mathbf{S}_h)^T(\mathbf{S} - \mathbf{S}_h)) d\Omega\right]^{1/2}$"

    for i, (Np_1d, sub_df) in enumerate(df.groupby("Np_1d")):
        if what == "stress_time":
            x = sub_df["total_time"]
        else:
            x = sub_df["h"]

        if what == "roi":
            y = 1.0 / sub_df["stress_norm"] / sub_df["total_time"]
        else:
            y = sub_df["stress_norm"]

        # Get averaged slope
        label = f"$p={Np_1d - 1}$"

        ax.loglog(
            x,
            y,
            "-o",
            label=label,
            lw=1.0,
            markeredgewidth=1.0,
            markersize=6.0,
            markeredgecolor="black",
            color=colors[i],
        )

    ymin, ymax = ax.get_ylim()
    v_off = -np.log10(ymax / ymin) * 0.03
    expand_logy_bottom(ax, frac=voffset_text * 0.07)

    # Annotate the slopes
    for Np_1d, sub_df in df.groupby("Np_1d"):
        if what == "stress_time":
            x = sub_df["total_time"]
        else:
            x = sub_df["h"]

        if what == "roi":
            y = 1.0 / sub_df["stress_norm"] / sub_df["total_time"]
        else:
            y = sub_df["stress_norm"]

        annotate_averaged_slope(ax, x, y, v_off * voffset)

    if what != "stress_time":
        # Remove all existing ticks
        ax.tick_params(axis="x", which="both", length=0, labelbottom=False)

        # Set new ticks with explicit positions and labels
        ax.set_xticks(df["h"].drop_duplicates())
        ax.set_xticklabels(
            df["h"].drop_duplicates().apply(lambda x: f"{x:.1e}"),
            rotation=45,
            ha="right",
        )

        # Add ticks
        ax.tick_params(
            axis="x", which="major", direction="in", length=3, labelbottom=True
        )

    ax.legend()
    ax.set_xlabel(xlabel)
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
    p.add_argument(
        "--what",
        nargs="*",
        default=["stress", "cpu_time", "roi", "stress_time", "cpu_time_breakdown"],
        choices=["stress", "cpu_time", "roi", "stress_time", "cpu_time_breakdown"],
    )
    p.add_argument("--csv", type=str)
    p.add_argument("--voffset", default=1.0, type=float, help="voffset scaler")
    p.add_argument(
        "--voffset_text", default=1.0, type=float, help="voffset_text scaler"
    )
    p.add_argument("--save-vtk", action="store_true")
    p.add_argument("--use-ersatz", action="store_true")
    p.add_argument("--ersatz-ratio", default=1e-6, type=float)
    p.add_argument("--nitsche-eta", default=1e8, type=float)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--Np_1d", nargs="*", default=[2, 4, 6], type=int)
    p.add_argument("--nxy-min", type=int, default=22)
    p.add_argument("--nxy-max", type=int, default=128)
    p.add_argument("--nxy-num", type=int, default=13)
    args = p.parse_args()

    # Sanity checks
    if args.physics == "elasticity-mms" and args.instance == "square":
        print(f"[Error] square is not implemented for elasticity-mms")
        exit(-1)
    if args.physics == "elasticity-interface" and args.instance == "square":
        print(f"[Error] square is not implemented for elasticity-interface")
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
    elif args.physics == "elasticity-interface":
        run_name = f"{args.physics}_energy_precision_{args.mesh}_nitscheeta_{args.nitsche_eta:.0e}"
    else:
        run_name = f"{args.physics}_energy_precision_{args.mesh}"

    if args.physics == "elasticity-bulk" and args.use_ersatz:
        run_name += f"_ersatz_{args.ersatz_ratio}"

    if args.smoke_test:
        run_name = "smoke_" + run_name

    if not os.path.isdir(run_name):
        os.mkdir(run_name)

    csv_try = os.path.join(run_name, f"{run_name}.csv")
    if args.csv:
        csv_try = args.csv

    if os.path.isfile(csv_try):
        df = pd.read_csv(csv_try)

    else:
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
            args.Np_1d,
            args.nxy_min,
            args.nxy_max,
            args.nxy_num,
        )
    print(df)

    for what in args.what:
        if what == "cpu_time":
            fig, _ = plot_cpu_time(df, args.voffset, args.voffset_text)
        elif what == "cpu_time_breakdown":
            fig, _ = plot_cpu_time_breakdown(df)
        else:
            if args.physics == "poisson":
                fig, _ = plot_poisson(df, args.voffset, args.voffset_text)
            elif args.physics == "elasticity-interface":
                fig, _ = plot_elasticity_interface(
                    df, what, args.voffset, args.voffset_text
                )
            else:
                fig, _ = plot_elasticity(df, what, args.voffset, args.voffset_text)

        fig_name = run_name
        if what != "stress":
            fig_name = what + "_" + fig_name

        fig.savefig(os.path.join(run_name, f"{fig_name}.pdf"))
        fig.savefig(os.path.join(run_name, f"{fig_name}.svg"))
