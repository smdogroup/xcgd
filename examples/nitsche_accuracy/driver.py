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


def get_poisson_l2_error_deprecated(prefix, cmd):
    subprocess.run(cmd, capture_output=True)

    with open(join(prefix, "sol.json")) as f:
        j = json.load(f)

    def compute_error(sol, sol_exact):
        return np.abs(
            np.linalg.norm(sol, ord=2) / np.linalg.norm(sol_exact, ord=2) - 1.0
        )

    sol = np.array(j["sol"])
    sol_exact = np.array(j["sol_exact"])
    lsf = np.array(j["lsf"])
    solq_bulk = np.array(j["solq_bulk"])
    solq_bulk_exact = np.array(j["solq_bulk_exact"])
    solq_bcs = np.array(j["solq_bcs"])
    solq_bcs_exact = np.array(j["solq_bcs_exact"])

    err_raw = compute_error(sol, sol_exact)
    err_interior = compute_error(sol[lsf < 0.0], sol_exact[lsf < 0.0])
    err_quad_bulk = compute_error(solq_bulk, solq_bulk_exact)
    err_quad_surf = compute_error(solq_bcs, solq_bcs_exact)

    return err_raw, err_interior, err_quad_bulk, err_quad_surf


def get_poisson_l2_error(prefix, cmd):
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

    return j["err_l2norm_bulk"], j["err_l2norm_bcs"]


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


def run_experiments(instance, Np_1d_list, nxy_list, nitsche_eta_list):
    df_data = {
        "Np_1d": [],
        "nxy": [],
        "h": [],
        "nitsche_eta": [],
        "err_l2norm_bulk": [],
        "err_l2norm_bcs": [],
    }

    for Np_1d in Np_1d_list:
        for nxy in nxy_list:
            for nitsche_eta in nitsche_eta_list:
                prefix = f"outputs_Np_{Np_1d}_nxy_{nxy}_nitsche_{nitsche_eta:.1e}"
                cmd = [
                    "./nitsche_accuracy",
                    f"--instance={instance}",
                    "--save-degenerate-stencils=0",
                    f"--Np_1d={Np_1d}",
                    f"--nxy={nxy}",
                    f"--prefix={prefix}",
                    f"--nitsche_eta={nitsche_eta}",
                ]

                t1 = time()
                (
                    err_l2norm_bulk,
                    err_l2norm_bcs,
                ) = get_poisson_l2_error(prefix, cmd)

                t2 = time()

                print(
                    f"Np_1d: {Np_1d:2d}, nxy: {nxy:4d}, nitsche_eta: {nitsche_eta:.2e}, execution time: {t2 - t1:.2f} s"
                )

                df_data["Np_1d"].append(Np_1d)
                df_data["nxy"].append(nxy)
                df_data["h"].append(2.0 / nxy)
                df_data["nitsche_eta"].append(nitsche_eta)

                df_data["err_l2norm_bulk"].append(err_l2norm_bulk)
                df_data["err_l2norm_bcs"].append(err_l2norm_bcs)

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


def plot(cases_df, xname):
    assert xname in ["h", "nitsche_eta"]

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
        for j, err_type in enumerate(["err_l2norm_bulk", "err_l2norm_bcs"]):
            df = cases_df[cases_df["Np_1d"] == Np_1d]

            x = df[xname]
            y = df[err_type]

            # Get averaged slope
            slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)

            label = f"$p={Np_1d - 1}, \Delta:{slope:.2f}$, {err_type}"
            ax.loglog(x, y, linestyles[j], color=colors[i], clip_on=False, label=label)

            print(label)

            # x0, x1 = x.iloc[-2:]
            # y0, y1 = y.iloc[-2:]
            # annotate_slope(ax, (x0, y0), (x1, y1))

    if xname == "h":
        ax.set_xlabel("Mesh size $h$")
    else:
        ax.set_xlabel(r"Nitsche parameter $\eta$")

    ax.set_ylabel(
        "Normalized relative solution error\n"
        + r"$|\dfrac{||u||_2^\text{CGD}}{||u||_2^\text{exact}} - 1|$"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    adjust_plot_lim(ax)

    return fig, ax


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.HelpFormatter)

    p.add_argument("--csv", type=str, default=None, help="case data csv")
    p.add_argument("--instance", default="circle", choices=["circle", "wedge"])
    p.add_argument(
        "--Np_1d",
        nargs="+",
        type=int,
        default=[2, 4, 6],
        help="list of Np_1d to use",
    )
    p.add_argument(
        "--nxy",
        nargs="+",
        default=[8, 16, 32, 64, 128],
        type=int,
        help="list of number of mesh elements per dimension",
    )
    p.add_argument(
        "--nitsche_eta",
        nargs="+",
        default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        type=float,
        help="list of Nitsche parameter",
    )

    p.add_argument(
        "--xname",
        default="h",
        choices=["h", "nitsche_eta"],
        help="which quantity to use for x axis of the plot",
    )

    args = p.parse_args()

    if args.csv is None:
        df = run_experiments(
            instance=args.instance,
            Np_1d_list=args.Np_1d,
            nxy_list=args.nxy,
            nitsche_eta_list=args.nitsche_eta,
        )
        df.to_csv("cases_data.csv")
    else:
        df = pd.read_csv(args.csv)
        drop_column = "Unnamed: 0"
        if drop_column in df.columns:
            df = df.drop(columns=[drop_column])
        df = df.sort_values(["Np_1d", "nxy"], ascending=[True, True]).reset_index(
            drop=True
        )

        # with pd.option_context("display.max_rows", None, "display.max_columns", None):
        #     print(df)
        #     exit()

    fig, ax = plot(df, args.xname)
    fig.savefig(f"nitsche_precision_x_{args.xname}.pdf")
    df.to_csv(f"nitsche_precision_x_{args.xname}.csv", index=False)
