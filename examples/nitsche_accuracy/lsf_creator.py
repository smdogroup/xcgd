from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import skfmm
from scipy.interpolate import interpn
import json
import argparse


def get_lsf_dof(nxy, imgfile, plot=True):
    image0 = io.imread(imgfile, as_gray=True)

    npx_h, npx_w = image0.shape

    if npx_w > npx_h:
        image = np.pad(image0, ((npx_w - npx_h, 0), (0, 0)), constant_values=1.0)
    else:
        image = np.pad(image0, ((0, 0), (npx_h - npx_w, 0)), constant_values=1.0)

    npx_1d = image.shape[0]

    # raw lsf
    phi_raw = 0.5 - (image < 0.9).astype(float)  # phi <= 0.0 is inside

    # Signed distance function
    phi = skfmm.distance(phi_raw)

    # Downsample
    points_1d = np.array([i / (npx_1d - 1.0) for i in range(npx_1d)])
    lsf_dof = interpn(
        (points_1d, points_1d),
        phi,
        [[ix / nxy, iy / nxy] for ix in range(nxy + 1) for iy in range(nxy + 1)],
        method="cubic",
    ).reshape(nxy + 1, nxy + 1)

    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(15.0, 5.0))
        for a in ax:
            a.axis("off")

        ax[0].imshow(image0)
        ax[0].set_title("original image")

        ax[1].imshow(phi_raw)
        ax[1].set_title("thresholded image")

        ax[2].imshow(lsf_dof)
        ax[1].set_title("signed distance function")

        plt.show()

    return np.flip(lsf_dof, axis=0).flatten()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--nxy", default=128)
    p.add_argument("--image", default="./images/stanford_dragon.png")
    p.add_argument("--plot", action="store_true")

    args = p.parse_args()

    nxy = args.nxy
    lsf_dof = get_lsf_dof(nxy=nxy, imgfile=args.image, plot=args.plot)

    with open(f"lsf_dof_nxy_{nxy}.json", "w") as f:
        json.dump({"nxy": nxy, "lsf_dof": lsf_dof.tolist()}, f, indent=2)
