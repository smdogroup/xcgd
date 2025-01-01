import os
from os.path import join
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import re
import pandas as pd
import seaborn as sns
import argparse


def remove_file_if_exists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def create_case_cfg(working_dir, cfg_template_path, case_cfg_path, h: float):
    with open(cfg_template_path, "r") as infile, open(case_cfg_path, "w") as outfile:
        for line in infile:
            line = re.sub(r"grad_check_fd_h.*\n", f"grad_check_fd_h = {h:.1e}\n", line)
            line = re.sub(
                r"check_grad_and_exit.*\n", f"check_grad_and_exit = true\n", line
            )
            line = re.sub(r"prefix.*\n", f'prefix = "{working_dir}/h_{h:.1e}"', line)
            outfile.write(line)


def execute(num_points: int):
    working_dir = "step_size_study"
    exec_name = "topo"
    cfg_template_name = "topo.cfg"

    # h_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    h_list = np.logspace(-3, -10, num_points)

    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    # Remove old files
    remove_file_if_exists(join(working_dir, exec_name))
    for cfg_path in glob.glob(join(working_dir, "*.cfg")):
        remove_file_if_exists(cfg_path)

    # Remove directories
    for item in os.listdir(working_dir):
        item_path = join(working_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # Launch cases
    df = []
    for h in tqdm(h_list):
        # Copy and modify case cfg
        case_cfg_path = join(working_dir, f"h_{h:.1e}.cfg")
        create_case_cfg(working_dir, cfg_template_name, case_cfg_path, h)

        result = subprocess.run(
            [join(".", exec_name), case_cfg_path],
            capture_output=True,
            text=True,
            check=True,
        )

        df_line = {"h": h}

        for line in result.stdout.split("\n"):
            m = re.search(r"(.*)FD.*Rel err:\s+([\d\.+\-e]+)", line)
            if m:
                name = m.group(1).strip()
                relerr = abs(float(m.group(2)))
                df_line[name] = relerr
        df.append(df_line)

    df = pd.DataFrame(df)
    df.to_csv(join(working_dir, "df.csv"))
    return df


def visualize(df):
    print(df)

    fig, ax = plt.subplots()

    for col in df.columns:
        if col != "h":
            ax.loglog(df["h"], df[col], "-o", label=col)

    ax.set_xlabel("h")
    ax.set_ylabel("rel. err.")
    ax.legend()
    ax.grid(which="both")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str)
    p.add_argument("--num-points", type=int, default=10)
    args = p.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv).drop(columns=["Unnamed: 0"])
    else:
        df = execute(args.num_points)

    visualize(df)
