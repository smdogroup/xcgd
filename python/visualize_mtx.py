from scipy.io import mmread
import argparse
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("mtx", nargs="*")
p.add_argument("--matshow", action="store_true")
p.add_argument("--check-symmetry", action="store_true")
p.add_argument("--check-symmetry-abs-tol", type=float, default=1e-10)
p.add_argument("--drop-zeros", action="store_true")
args = p.parse_args()

mat_list = []

for mtx in args.mtx:
    fig, ax = plt.subplots()
    mat = mmread(mtx)
    mat_list.append(mat)
    if args.matshow:
        ax.matshow(mat.todense())
    else:
        if args.drop_zeros:
            mat.eliminate_zeros()
        ax.spy(mat)

# Compute difference
if len(mat_list) == 2:
    mat1 = mat_list[0]
    mat2 = mat_list[1]

    print("max entry of mat 1: %20.10f" % np.max(np.abs(mat1)))
    print("max entry of mat 2: %20.10f" % np.max(np.abs(mat2)))
    print("max entry diff: %20.10f" % np.max(np.abs(mat1 - mat2)))
    print()
    print("min entry of mat 1: %20.10f" % np.min(np.abs(mat1)))
    print("min entry of mat 2: %20.10f" % np.min(np.abs(mat2)))
    print("min entry diff: %20.10f" % np.min(np.abs(mat1 - mat2)))

    fig, ax = plt.subplots()
    ax.matshow((mat1 - mat2).todense())

# Check symmetry
if args.check_symmetry:
    print("max(mat - mat.T): %20.10e" % (mat - mat.T).max())
    print("min(mat - mat.T): %20.10e" % (mat - mat.T).min())

    # indices for upper-triangular asymmetrical entries in the matrix
    indices = np.nonzero((mat - mat.T) > args.check_symmetry_abs_tol)
    indices = [(i, j) for (i, j) in zip(*indices) if i <= j]

    A = mat.todense()
    for i, j in indices:
        print(
            f"A[{i}, {j}]: {A[i, j]:20.10e}, A[{j}, {i}]: {A[j, i]:20.10e}, diff: {A[i,j] - A[j, i] : 20.10e}"
        )


plt.show()
