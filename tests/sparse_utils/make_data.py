import numpy as np
from scipy.sparse import random, bsr_matrix, eye, linalg
import matplotlib.pyplot as plt
import json


def create_spd_csr(nrows=20):
    np.random.seed(0)

    # Generate random block sparse matrix
    mat = random(
        nrows,
        nrows,
        density=0.1,
        format="csr",
        dtype=np.float64,
    )

    mat = mat + mat.T
    emin = min(linalg.eigsh(mat, which="SA")[0])
    mat -= 1.5 * emin * eye(nrows)

    nnz = mat.nnz
    cols = mat.indices
    rowp = mat.indptr
    vals = mat.data

    return mat, nnz, cols, rowp, vals, 1


def create_spd_bsr(nbrows=4, block_size=2):
    _, nnz, cols, rowp, _, _ = create_spd_csr(nrows=nbrows)
    data = np.random.rand(nnz, block_size, block_size)

    # Construct a positive symmetric definite matrix
    mat = bsr_matrix((data, cols, rowp))
    mat = mat + mat.T
    emin = min(linalg.eigsh(mat, which="SA")[0])
    mat -= 1.5 * emin * eye(nbrows * block_size)

    cols = mat.indices
    rowp = mat.indptr
    vals = mat.data.flatten()
    nnzb = int(rowp[-1])

    return mat, nnzb, cols, rowp, vals, block_size


def create_dict_entry(mat_creator):
    mat, nnz, cols, rowp, vals, block_size = mat_creator()

    rhs = np.random.rand(mat.shape[0])

    sol = linalg.spsolve(mat.tocsr(), rhs)

    return {
        "block_size": block_size,
        "nrows": int(len(rowp) - 1),
        "nnz": nnz,
        "cols": cols.tolist(),
        "rowp": rowp.tolist(),
        "vals": vals.tolist(),
        "rhs": rhs.tolist(),
        "sol": sol.tolist(),
    }


def test_bsr():
    np.set_printoptions(precision=3)
    mat, nnzb, cols, rowp, vals, block_size = create_spd_bsr(nbrows=2, block_size=2)
    print(mat.toarray())

    print("vals:")
    print(vals)
    exit()


if __name__ == "__main__":
    # test_bsr()

    np.random.seed(0)

    inputs = {
        "csr": create_dict_entry(lambda: create_spd_csr(nrows=10)),
        "bsr": create_dict_entry(lambda: create_spd_bsr(nbrows=10, block_size=3)),
    }

    with open("data.json", "w") as f:
        json.dump(inputs, f, indent=2)
