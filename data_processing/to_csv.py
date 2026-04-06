#!/usr/bin/env python3
"""
Convert ROS2 bag files to CSVs, extracting messages from /pf/pose/odom.

No ROS2 installation required — uses the pure-Python 'rosbags' library.
Install with:  pip install rosbags

Expected directory structure:
    lab5/
    ├── bags/                (contains rosbag directories)
    └── data_processing/
        └── bags_to_csv.py   (this script)

Output:
    lab5/data_processing/csvs/  (one CSV per bag)
"""

import csv
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores


TOPIC = "/pf/pose/odom"
SCRIPT_DIR = Path(__file__).resolve().parent
BAGS_DIR = SCRIPT_DIR.parent / "bags"
OUTPUT_DIR = SCRIPT_DIR / "csvs"

CSV_HEADER = [
    "timestamp_sec",
    "timestamp_nanosec",
    "frame_id",
    "pos_x",
    "pos_y",
    "orient_x",
    "orient_y",
    "orient_z",
    "orient_w",
]


def convert_bag(bag_path: Path, output_dir: Path, typestore):
    """Read a single bag and write /pf/pose/odom messages to a CSV."""
    with Reader(bag_path) as reader:
        # Check if topic exists
        topic_names = [c.topic for c in reader.connections]
        if TOPIC not in topic_names:
            print(f"  [SKIP] Topic '{TOPIC}' not found in {bag_path.name}")
            return

        # Get the connection(s) for our topic
        connections = [c for c in reader.connections if c.topic == TOPIC]
        count = 0
        csv_path = output_dir / (bag_path.name + ".csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

            for conn, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                h = msg.header
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                writer.writerow([
                    h.stamp.sec,
                    h.stamp.nanosec,
                    h.frame_id,
                    p.x, p.y,
                    o.x, o.y, o.z, o.w,
                ])
                count += 1

        print(f"  [OK]   {bag_path.name} -> {csv_path.name}  ({count} messages)")


def find_bag_paths(bags_dir: Path):
    """Yield each bag directory found under bags_dir."""
    for entry in sorted(bags_dir.iterdir()):
        if entry.is_dir() and (
            any(entry.glob("*.db3")) or any(entry.glob("*.mcap"))
        ):
            yield entry


def main():
    if not BAGS_DIR.is_dir():
        print(f"Error: bags directory not found at {BAGS_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the ROS2 Humble type store (works for Iron/Jazzy too)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    bag_paths = list(find_bag_paths(BAGS_DIR))
    if not bag_paths:
        print(f"No bag files found in {BAGS_DIR}")
        return

    print(f"Found {len(bag_paths)} bag(s) in {BAGS_DIR}\n")

    for bp in bag_paths:
        print(f"Processing: {bp.name}")
        try:
            convert_bag(bp, OUTPUT_DIR, typestore)
        except Exception as e:
            print(f"  [ERR]  {bp.name}: {e}")

    print(f"\nDone. CSVs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()