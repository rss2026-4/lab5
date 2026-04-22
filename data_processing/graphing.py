import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

csv_dir = Path("csvs")
csv_files = sorted(csv_dir.glob("*.csv"))

# ---- EDIT THIS LIST ----
CSV_NAMES = [
    # "noise.15.csv",
    # "noise.15_tf.csv",
    # "noise1.0_tf.csv",
    # "noise1.0.csv",
    # "noise2.0_tf.csv",
    # "noise2.0.csv",
    # "noise3.0_tf.csv",
    # "noise3.0.csv",
    "const_noise.csv"
]
# -------------------------

csv_dir = Path("csvs")
bags = {}
for name in CSV_NAMES:
    f = f"C:\\Users\\coold\\racecar_docker\\home\\racecar_ws\\lab05\\data_processing\\csvs\\{name}"

    df = pd.read_csv(f)
    df["time"] = df["timestamp_sec"] + df["timestamp_nanosec"] * 1e-9
    df["time"] -= df["time"].iloc[0]
    quats = df[["orient_x", "orient_y", "orient_z", "orient_w"]].values
    df["yaw"] = Rotation.from_quat(quats).as_euler("xyz")[:, 2]
    bags[name] = df

print(f"Loaded {len(bags)}/{len(CSV_NAMES)} bag(s)")

def plot_bag_sim(name, title=None):
    df = pd.read_csv(f"C:\\Users\\coold\\racecar_docker\\home\\racecar_ws\\lab05\\data_processing\\csvs\\{name}.csv")
    df_tf = pd.read_csv(f"C:\\Users\\coold\\racecar_docker\\home\\racecar_ws\\lab05\\data_processing\\csvs\\{name}_tf.csv")
    df["time"] = df["timestamp_sec"] + df["timestamp_nanosec"] * 1e-9
    df["time"] -= df["time"].iloc[0]
    quats = df[["orient_x", "orient_y", "orient_z", "orient_w"]].values
    df["yaw"] = Rotation.from_quat(quats).as_euler("xyz")[:, 2]
    quats_tf = df_tf[["orient_x", "orient_y", "orient_z", "orient_w"]].values
    df_tf["yaw"] = Rotation.from_quat(quats_tf).as_euler("xyz")[:, 2]

    display_title = title if title else name

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    axes.plot(df["pos_x"], df["pos_y"], linewidth=1.5, label = 'estimated path')
    axes.plot(df_tf["pos_x"], df_tf["pos_y"], linewidth=1.5, label = 'ground truth path' )
    axes.scatter(df["pos_x"].iloc[0], df["pos_y"].iloc[0], marker="o", s=80, c="green", label="start", zorder=5)
    axes.scatter(df["pos_x"].iloc[-1], df["pos_y"].iloc[-1], marker="x", s=80, c="green", label="end", zorder=5)
    axes.scatter(df_tf["pos_x"].iloc[0], df_tf["pos_y"].iloc[0], marker="o", s=80, c="blue", label="start (truth)", zorder=5)
    axes.scatter(df_tf["pos_x"].iloc[-1], df_tf["pos_y"].iloc[-1], marker="x", s=80, c="blue", label="end (truth)", zorder=5)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("Path (x vs y)")
    axes.set_aspect("equal")
    axes.legend()
    axes.grid(True, alpha=0.3)

    x_truth = np.array(df_tf['pos_x'])
    x_truth = x_truth[:(len(x_truth)-1)]
    x_est = np.array(df['pos_x'])[::2]

    y_truth = np.array(df_tf['pos_y'])
    y_truth = y_truth[:(len(y_truth)-1)]
    y_est = np.array(df['pos_y'])[::2]
    # print(x_truth.shape)
    # print(x_est.shape)
    # print(x_truth-x_est)
    x_err = np.mean(x_truth-x_est)
    print(f"{x_err=}")
    y_err = np.mean(y_truth-y_est)
    print(f"{y_err=}")

    err = (x_err+y_err)/2
    print(err)

    # ax_x = axes[1]
    # ax_y = ax_x.twinx()
    # ax_x.plot(df["time"], df["pos_x"], label="x", color="tab:blue")
    # ax_x.plot(df["time"], df["pos_y"], label="y", color="tab:orange")
    # ax_x.set_xlabel("time (s)")
    # ax_x.set_ylabel("position (m)")
    # ax_y.plot(df["time"], np.degrees(df["yaw"]), label="yaw", color="tab:green", alpha=0.7)
    # ax_y.set_ylabel("yaw (°)")
    # ax_x.legend(loc="upper left")
    # ax_y.legend(loc="upper right")
    # axes[1].set_title("Position & Heading over Time")
    # ax_x.grid(True, alpha=0.3)

    fig.suptitle(display_title, fontsize=14)
    plt.tight_layout()
    # plt.show()

# plot_bag_sim("noise.15", "estimate vs ground truth (noise = 0.15)")
# plot_bag_sim("noise1.0", "estimate vs ground truth (noise = 1.0)")
# plot_bag_sim("noise2.0", "estimate vs ground truth (noise = 2.0)")
# plot_bag_sim("noise3.0", "estimate vs ground truth (noise = 3.0)")
plot_bag_sim("const_noise", "estimate vs ground truth (noise = 3.0)")