import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess

cmd = ["./surface_integral"]
subprocess.run(cmd)

with open("quadrature_data.json", "r") as f:
    d = json.load(f)

ws = np.array(d["quad_weights"])
pts = np.array(d["quad_points"])
ns = np.array(d["quad_norms"])


fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.scatter(
    pts[:, 0], pts[:, 1], color="red", s=20, label="quadrature points", clip_on=False
)
ax.quiver(
    pts[:, 0],
    pts[:, 1],
    ns[:, 0],
    ns[:, 1],
    label="quadrature point normals",
    clip_on=False,
)

ax.set_aspect("equal", "box")
ax.legend(frameon=False)

plt.show()
