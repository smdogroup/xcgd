import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import scienceplots

# # Get ggplot colors
colors = plt.style.library["ggplot"]["axes.prop_cycle"].by_key()["color"]

plt.style.use(["science"])

p = argparse.ArgumentParser()
p.add_argument("logfile", type=str)

args = p.parse_args()

with open(args.logfile, "r") as f:
    df = pd.read_csv(f, delimiter=r"\s+", skip_blank_lines=True)

# Remove extra header lines
df = df[df["major"] != "major"]
df = df.apply(lambda col: pd.to_numeric(col, errors="ignore"))

df["group"] = (df["major"] == 0).cumsum()
df = df[df["group"] == df["group"].max()]
df = df.drop(columns=["group"])
df = df.reset_index(drop=True)


fig, ax = plt.subplots(figsize=(4.8, 3.6))

obj_color_index = 2
ax.plot(
    df["major"],
    df["obj"],
    "-",
    lw=1.0,
    color=colors[obj_color_index],
)

# vol_color_index = 3
# ax2 = ax.twinx()
# ax2.plot(
#     df["major"],
#     df[r"vol(%)"],
#     "-",
#     lw=1.0,
#     color=colors[vol_color_index],
# )
# ax2.set_ylabel(r"Volume (\%)")

ax.set_xlabel("Optimization Iteration")
ax.set_ylabel("Objective Value")


plt.savefig(os.path.join(os.path.dirname(args.logfile), "optimization_history.svg"))
