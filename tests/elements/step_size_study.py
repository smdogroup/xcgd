import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--physics", default="Elasticity", choices=["Poisson", "Elasticity", "Load"]
    )
    args = p.parse_args()

    cmd = [f'./test_element_utils --gtest_filter="*{args.physics}*"']
    print(cmd)
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, shell=True
    )

    df = []
    for line in result.stdout.split("\n"):
        m = re.search(
            r"Np_1d:\s+(\d+).*dh:\s+([\d\.+\-e]+).*FD.*Rel err:\s+([\d\.+\-e]+)", line
        )
        if m:
            df.append(
                {
                    "Np_1d": int(m.group(1)),
                    "dh": float(m.group(2)),
                    "relerr": float(m.group(3)),
                }
            )

    df = pd.DataFrame(df)

    print(df)

    fig, ax = plt.subplots()

    for Np_1d, d in df.groupby("Np_1d"):
        ax.loglog(d["dh"], d["relerr"], "-o", label=f"Np_1d={Np_1d}")
    ax.set_xlabel("h")
    ax.set_ylabel("rel. err.")
    ax.legend()
    ax.grid(which="both")
    plt.show()
    fig.savefig(f"step_size_study_{args.physics}.pdf")
