import numpy as np
from scipy.io import mmread
import argparse


p = argparse.ArgumentParser()
p.add_argument("mtx", type=str)
args = p.parse_args()

mat = mmread(args.mtx)

print("matrix size: %d by %d" % (mat.shape[0], mat.shape[1]))
print("nnz: %d (%.2f %%)" % (mat.nnz, mat.nnz / (mat.shape[0] * mat.shape[1]) * 100.0))

mat = mat.todense()

print("mat(A):", np.max(mat))
print("mat(A):", np.min(mat))
print("Forbenius norm:", np.linalg.norm(mat))
print("max(A - A.T):", np.max(mat - mat.T))
print("min(A - A.T):", np.min(mat - mat.T))
print("condition number: %20.10e" % np.linalg.cond(mat))
