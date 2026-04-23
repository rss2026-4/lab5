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
    f = csv_dir / name

    df = pd.read_csv(f)
    df["time"] = df["timestamp_sec"] + df["timestamp_nanosec"] * 1e-9
    df["time"] -= df["time"].iloc[0]
    quats = df[["orient_x", "orient_y", "orient_z", "orient_w"]].values
    df["yaw"] = Rotation.from_quat(quats).as_euler("xyz")[:, 2]
    bags[name] = df

print(f"Loaded {len(bags)}/{len(CSV_NAMES)} bag(s)")

def plot_bag_sim(name, title=None):
    df = pd.read_csv(csv_dir / f"{name}.csv")
    df_tf = pd.read_csv(csv_dir / f"{name}_tf.csv")
    t_est = (df["timestamp_sec"] + df["timestamp_nanosec"] * 1e-9).values
    t_truth = (df_tf["timestamp_sec"] + df_tf["timestamp_nanosec"] * 1e-9).values
    t0 = min(t_est[0], t_truth[0])
    t_est -= t0
    t_truth -= t0

    quats = df[["orient_x", "orient_y", "orient_z", "orient_w"]].values
    df["yaw"] = Rotation.from_quat(quats).as_euler("xyz")[:, 2]
    quats_tf = df_tf[["orient_x", "orient_y", "orient_z", "orient_w"]].values
    df_tf["yaw"] = Rotation.from_quat(quats_tf).as_euler("xyz")[:, 2]

    x_est_interp = np.interp(t_truth, t_est, df["pos_x"].values)
    y_est_interp = np.interp(t_truth, t_est, df["pos_y"].values)
    x_truth = df_tf["pos_x"].values
    y_truth = df_tf["pos_y"].values

    errors = np.sqrt((x_truth - x_est_interp)**2 + (y_truth - y_est_interp)**2)
    rmse = np.sqrt(np.mean(errors**2))
    mean_err = np.mean(errors)

    display_title = title if title else name

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(x_est_interp, y_est_interp, linewidth=1.5, label="estimated path")
    axes[0].plot(x_truth, y_truth, linewidth=1.5, label="ground truth path")
    axes[0].scatter(x_est_interp[0], y_est_interp[0], marker="o", s=80, c="green", label="start (est)", zorder=5)
    axes[0].scatter(x_est_interp[-1], y_est_interp[-1], marker="x", s=80, c="green", label="end (est)", zorder=5)
    axes[0].scatter(x_truth[0], y_truth[0], marker="o", s=80, c="blue", label="start (truth)", zorder=5)
    axes[0].scatter(x_truth[-1], y_truth[-1], marker="x", s=80, c="blue", label="end (truth)", zorder=5)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("Path (x vs y)")
    axes[0].set_aspect("equal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_truth, errors, linewidth=1.0)
    axes[1].axhline(mean_err, color="red", linestyle="--", label=f"mean = {mean_err:.3f} m")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("euclidean error (m)")
    axes[1].set_title(f"Error over time (RMSE = {rmse:.3f} m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    print(f"[{name}] RMSE = {rmse:.4f} m, Mean Error = {mean_err:.4f} m")

    fig.suptitle(display_title, fontsize=14)
    plt.tight_layout()

plot_bag_sim("noise.15", "estimate vs ground truth (noise = 0.15)")
plot_bag_sim("noise1.0", "estimate vs ground truth (noise = 1.0)")
plot_bag_sim("noise2.0", "estimate vs ground truth (noise = 2.0)")
plot_bag_sim("noise3.0", "estimate vs ground truth (noise = 3.0)")
plt.show()