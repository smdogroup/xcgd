import numpy as np
from scipy.io import mmread
import matplotlib.pyplot as plt
import argparse
from scipy.sparse import linalg


p = argparse.ArgumentParser()
p.add_argument("mtx", type=str)
p.add_argument("--N", type=int, default=20)
p.add_argument("--all-eigenvalues", action="store_true")
args = p.parse_args()

mat = mmread(args.mtx)
if args.all_eigenvalues:
    mat = mat.todense()
    e, v = np.linalg.eigh(mat)
else:
    e, v = linalg.eigsh(mat, args.N, which="SA")

print("min eig: %20.10e" % e[0])

plt.plot(e, "-o")
plt.show()
