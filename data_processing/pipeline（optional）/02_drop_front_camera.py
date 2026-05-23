#!/usr/bin/env python3
"""
Copy HDF5 episodes, dropping the front_image_1 camera stream.

Usage:
    python3 02_drop_front_camera.py src dst
    python3 02_drop_front_camera.py path/to/raw_dir        path/to/no_front_dir
    python3 02_drop_front_camera.py path/to/episode_0.hdf5 path/to/episode_0_nf.hdf5
"""

import argparse
import glob
import os
import sys

import h5py


def drop_front(src: str, dst: str) -> None:
    with h5py.File(src, "r") as f:
        qpos = f["observations/qpos"][:]
        action = f["action"][:]
        exterior = f["observations/images/exterior_image_1_left"][:]
        wrist = f["observations/images/wrist_image_left"][:]
        attrs = dict(f.attrs)

    with h5py.File(dst, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.create_dataset("observations/qpos", data=qpos, compression="gzip")
        f.create_dataset("action", data=action, compression="gzip")
        f.create_dataset("observations/images/exterior_image_1_left", data=exterior, compression="gzip")
        f.create_dataset("observations/images/wrist_image_left", data=wrist, compression="gzip")


def collect_files(src: str) -> list[str]:
    if os.path.isfile(src):
        return [src]
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.hdf5")))
        if not files:
            sys.exit(f"No .hdf5 files found in: {src}")
        return files
    sys.exit(f"Path does not exist: {src}")


def main():
    parser = argparse.ArgumentParser(description="Copy episodes, dropping front_image_1.")
    parser.add_argument("src", help="Single .hdf5 file or directory containing *.hdf5 files")
    parser.add_argument("dst", help="Output file (single-file mode) or directory (batch mode)")
    args = parser.parse_args()

    files = collect_files(args.src)
    single = os.path.isfile(args.src)

    if single:
        drop_front(files[0], args.dst)
        print(f"  WROTE {args.dst}")
    else:
        os.makedirs(args.dst, exist_ok=True)
        print(f"Dropping front camera from {len(files)} episode(s)  →  {args.dst}")
        for path in files:
            name = os.path.basename(path)
            drop_front(path, os.path.join(args.dst, name))
            print(f"  WROTE {name}")
    print("Done.")


if __name__ == "__main__":
    main()
