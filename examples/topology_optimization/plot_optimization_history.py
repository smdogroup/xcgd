import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

p = argparse.ArgumentParser()
p.add_argument("logfile", type=str)
p.add_argument("--columns", nargs="+", default=["comp"])

args = p.parse_args()

with open(args.logfile, "r") as f:
    df = pd.read_csv(f, delimiter=r"\s+", skip_blank_lines=True)

# Remove extra header lines
df = df[df["iter"] != "iter"]
df = df.apply(lambda col: pd.to_numeric(col, errors="ignore"))

df["group"] = (df["iter"] == 0).cumsum()
df = df[df["group"] == df["group"].max()]
df = df.drop(columns=["group"])
df = df.reset_index(drop=True)


fig, ax = plt.subplots()
for col in args.columns:
    ax.plot(df["iter"], df[col], label=col)

ax.legend()
ax.grid()
ax.set_xlabel("Iter")

plt.show()
