"""
Usage:
./test_robust_projection --gtest_filter="apps.visualizeRobustProjection" | grep "proj:" | python plot_robust_projection.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


data = np.array([list(map(float, l.split()[1:3])) for l in sys.stdin])

plt.plot(data[:, 0], data[:, 1], "-", color="black")
ax = plt.gca()
ax.set_xlabel("raw")
ax.set_ylabel("projected")

ax.grid()
plt.show()
