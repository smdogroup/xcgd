import numpy as np
from scipy.io import mmread
import matplotlib.pyplot as plt
import argparse


p = argparse.ArgumentParser()
p.add_argument("mtx", type=str)
args = p.parse_args()

mat = mmread(args.mtx).todense()
e, v = np.linalg.eigh(mat)
plt.plot(e, "-o")
plt.show()
