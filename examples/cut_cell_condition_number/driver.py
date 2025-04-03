import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import mmread
from os import path
import os
import json
import matplotlib.patches as patches
import argparse

import scienceplots

# # Get ggplot colors
colors = plt.style.library["ggplot"]["axes.prop_cycle"].by_key()["color"]

plt.style.use(["science"])


def plot_mesh(prefix, Np_1d: int, nxy: int, use_finite_cell_mesh: bool, cell: int):

    cmd = [
        "./compare_cm_fcm",
        f"--Np_1d={Np_1d}",
        f"--nxy={nxy}",
        f"--prefix={prefix}",
        f"--use-finite-cell-mesh={int(use_finite_cell_mesh)}",
    ]

    with open(os.path.join(prefix, "mesh.json")) as f:
        j = json.load(f)

    # Extract and convert information from json
    nxy = j["nxy"]
    L = j["L"]
    lsf_dof = np.array(j["lsf_dof"]).reshape(nxy + 1, nxy + 1)
    cell_verts = {l[0]: l[1] for l in j["cell_verts"]}

    # Get a set of vert indices
    active_verts = {v for verts in cell_verts.values() for v in verts}

    lw = 1.0
    markersize = 15

    # Derived parameters
    hx = L / nxy
    hy = L / nxy

    num_points_x = nxy + 1
    num_points_y = nxy + 1
    # Create the x and y coordinates of the grid
    x = np.linspace(0.0, L, num_points_x)
    y = np.linspace(0.0, L, num_points_y)

    # Create the meshgrid
    X, Y = np.meshgrid(x, y)

    # Find dof nodes
    Xdof = []
    Ydof = []

    for vert in active_verts:

        ix = vert % (nxy + 1)
        iy = vert // (nxy + 1)

        Xdof.append(ix * L / nxy)
        Ydof.append(iy * L / nxy)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    # Plot the mesh
    ax.plot(X, Y, "-", color="grey", lw=lw, zorder=5, clip_on=False)
    ax.plot(X.T, Y.T, "-", color="grey", lw=lw, zorder=5, clip_on=False)

    # Plot the cut boundary
    ax.contour(X, Y, lsf_dof, levels=[0.0], colors="blue", linewidths=lw)

    # Plot active nodes
    ax.scatter(
        Xdof,
        Ydof,
        facecolor="grey",
        edgecolor="black",
        s=markersize,
        lw=lw,
        zorder=10,
        clip_on=False,
        label="active CGD dof nodes",
    )

    # Plot the chosen cell
    ie = cell // nxy
    je = cell % nxy
    rectangle = patches.Rectangle(
        (X[ie, je], Y[ie, je]),
        hx,
        hy,
        edgecolor="none",
        facecolor="blue",
        alpha=0.3,
    )
    ax.add_patch(rectangle)

    # Plot stencil for an element
    Xdof_elem = []
    Ydof_elem = []

    for vert in cell_verts[cell]:

        ix = vert % (nxy + 1)
        iy = vert // (nxy + 1)

        Xdof_elem.append(ix * L / nxy)
        Ydof_elem.append(iy * L / nxy)

    ax.scatter(
        Xdof_elem,
        Ydof_elem,
        facecolor="red",
        edgecolor="black",
        s=markersize,
        lw=lw,
        zorder=20,
        clip_on=False,
        label="stencil nodes for an element",
    )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.axis("off")

    handles, labels = ax.get_legend_handles_labels()
    handles.insert(1, plt.Line2D([0], [0], color="grey", lw=lw))
    labels.insert(1, "ground grid")
    handles.insert(3, plt.Line2D([0], [0], color="blue", lw=lw))
    labels.insert(3, "level set boundary")

    l = ax.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        bbox_transform=fig.transFigure,
        fontsize=15,
    )
    l.get_frame().set_edgecolor("none")
    l.get_frame().set_facecolor("none")

    # plt.show()
    fig.savefig(os.path.join(prefix, f"mesh_stencil.pdf"))

    plt.close()


def sweep(
    index,
    run_name,
    ax,
    Np_1d=2,
    exp_max=-1.0,
    exp_min=-5.0,
    num_pts=10,
    nelems=33,
    nitsche_eta=1e5,
    use_finite_cell_mesh=False,
    ersatz="none",
):

    if not os.path.isdir(run_name):
        os.mkdir(run_name)

    n = nelems
    l = 1.0
    h = l / n
    r0_max = 3.0 * h * 2.0**0.5
    p = Np_1d - 1

    delta = []
    cond = []

    for i, d in enumerate(tqdm(np.logspace(exp_max, exp_min, num_pts))):
        r = r0_max - d * h
        prefix = os.path.join(
            run_name,
            "cond_study_p_%s_%d_%d"
            % (
                "fcell" if use_finite_cell_mesh else "cut",
                p,
                i,
            ),
        )
        cmd = [
            "./condition_number",
            "--Np_1d=%d" % Np_1d,
            "--l=%.1f" % l,
            "--n=%d" % n,
            "--x0=%.1f" % l,
            "--y0=0.0",
            "--r=%.10f" % r,
            f"--nitsche-eta={nitsche_eta}",
            "--prefix=%s" % prefix,
            "--use-finite-cell-mesh=%d" % int(use_finite_cell_mesh),
            "--ersatz-method=%s" % ersatz,
        ]
        output = subprocess.run(cmd, capture_output=True)

        plot_mesh(
            prefix,
            Np_1d=Np_1d,
            nxy=n,
            use_finite_cell_mesh=use_finite_cell_mesh,
            cell=27,
        )

        delta.append(d)
        cond.append(
            np.linalg.cond(mmread(path.join(prefix, "stiffness_matrix.mtx")).todense())
        )

    ax.loglog(
        delta,
        cond,
        "-o",
        label=("p=%d" % p),
        clip_on=False,
        zorder=100,
        lw=1.0,
        markeredgewidth=1.0,
        markersize=6.0,
        markeredgecolor="black",
        color=colors[index],
    )

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", nargs="*", default=["fc", "cut"], type=str)
    p.add_argument("--Np_1d", nargs="*", default=[2, 4, 6], type=int)
    p.add_argument(
        "--ersatz",
        nargs="*",
        default=["none", "direct", "nitsche"],
        choices=["none", "direct", "nitsche"],
    )
    p.add_argument("--nitsche-eta", default=1e5, type=float)

    args = p.parse_args()

    for use_finite_cell_mesh in [m == "fc" for m in args.mesh]:
        for ersatz in args.ersatz:
            run_name = f'mesh_{"fc" if use_finite_cell_mesh else "cut"}_ersatz_{ersatz}'
            if ersatz == "nitsche":
                run_name += f"_{args.nitsche_eta:.0e}"

            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            for index, Np_1d in enumerate(args.Np_1d):
                sweep(
                    index=index,
                    run_name=run_name,
                    ax=ax,
                    Np_1d=Np_1d,
                    exp_max=-1.0,
                    exp_min=-5.0,
                    num_pts=10,
                    nelems=10,
                    nitsche_eta=args.nitsche_eta,
                    use_finite_cell_mesh=use_finite_cell_mesh,
                    ersatz=ersatz,
                )

            ax.set_xlabel(r"Cut-cell Ratio $\alpha$")
            ax.set_ylabel(r"Condition Number $\kappa(K)$")

            ax.invert_xaxis()
            ax.legend()
            plt.savefig(
                os.path.join(
                    run_name,
                    f"condition_number_study_{'fcell' if use_finite_cell_mesh else 'cut'}.pdf",
                )
            )
