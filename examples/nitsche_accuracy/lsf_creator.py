from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import skfmm
from scipy.interpolate import interpn
import json


def get_lsf_dof(nxy=128, plot=False):
    image = io.imread("stanford_dragon.png", as_gray=True)
    image = np.pad(
        image, ((image.shape[1] - image.shape[0], 0), (0, 0)), constant_values=1.0
    )
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
        plt.imshow(lsf_dof)
        plt.show()

    return np.flip(lsf_dof, axis=0).flatten()


if __name__ == "__main__":
    nxy = 128
    lsf_dof = get_lsf_dof(nxy=nxy)

    with open(f"lsf_dof_nxy_{nxy}.json", "w") as f:
        json.dump({"nxy": nxy, "lsf_dof": lsf_dof.tolist()}, f, indent=2)
