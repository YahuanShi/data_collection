#!/usr/bin/env python3
"""
Convert UR5 HDF5 episode dataset to Zarr format for Diffusion Policy training.

Reads all episode_N.hdf5 files from a dataset directory (written by
recorder/episode_recorder.py) and concatenates them
into a single Zarr store in the ReplayBuffer layout that Diffusion Policy
expects.

Source HDF5 layout (per episode):
    /observations/images/exterior_image_1_left  (T, H, W, 3) uint8
    /observations/images/wrist_image_left        (T, H, W, 3) uint8
    /observations/qpos                           (T, 7)       float64
    /action                                      (T, 7)       float64

Output Zarr layout:
    data/
        exterior_image   (T_total, H, W, 3)  uint8
        wrist_image      (T_total, H, W, 3)  uint8   [omit with --no-wrist]
        robot_qpos       (T_total, 7)         float32
        eef_pose         (T_total, 6)         float32  [omit with --no-eef or if absent in source]
        action           (T_total, 7)         float32
    meta/
        episode_ends     (n_episodes,)        int64   cumulative end indices

Usage:
    python data_processing/convert_hdf5_to_zarr.py \\
        --raw-dir  dataset/raw/ur5_dataset_20260415 \\
        --zarr-out dataset/zarr/ur5_dataset_20260415.zarr

    # Optionally include wrist camera or drop it:
    python data_processing/convert_hdf5_to_zarr.py \\
        --raw-dir dataset/raw/ur5_dataset_20260415 \\
        --zarr-out dataset/zarr/ur5_dataset_20260415.zarr \\
        --no-wrist

    # Optionally resize images (saves disk space):
    python data_processing/convert_hdf5_to_zarr.py \\
        --raw-dir dataset/raw/ur5_dataset_20260415 \\
        --zarr-out dataset/zarr/ur5_dataset_20260415.zarr \\
        --image-size 224
"""

import argparse
from pathlib import Path
import shutil

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import zarr


def center_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Square center-crop then resize to (size, size)."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def convert(
    raw_dir: Path,
    zarr_out: Path,
    image_size: int | None,
    include_wrist: bool,
    include_eef: bool,
    overwrite: bool,
) -> None:
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {raw_dir}")
    print(f"Found {len(hdf5_files)} episode(s) in {raw_dir}")

    if zarr_out.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {zarr_out}\nUse --overwrite to replace it.")
        print(f"Removing existing zarr at {zarr_out}")
        shutil.rmtree(zarr_out)

    # ── Detect image shape and optional fields from first episode ────────────
    with h5py.File(hdf5_files[0], "r") as f:
        raw_h, raw_w = f["/observations/images/exterior_image_1_left"].shape[1:3]
        has_eef = "/observations/eef_pose" in f
    out_h = image_size if image_size else raw_h
    out_w = image_size if image_size else raw_w
    print(f"Image: {raw_h}x{raw_w} → stored as {out_h}x{out_w}")
    if include_eef and not has_eef:
        print("Warning: --no-eef not set but eef_pose not found in dataset; skipping eef_pose.")
        include_eef = False

    # ── Create zarr store ─────────────────────────────────────────────────────
    store = zarr.DirectoryStore(str(zarr_out))
    root = zarr.group(store, overwrite=True)
    data_grp = root.require_group("data")
    meta_grp = root.require_group("meta")

    img_chunk = (1, out_h, out_w, 3)
    compressor = zarr.Blosc(cname="lz4", clevel=5)

    ext_arr = data_grp.zeros(
        "exterior_image", shape=(0, out_h, out_w, 3), chunks=img_chunk, dtype="uint8", compressor=compressor
    )
    wrist_arr = None
    if include_wrist:
        wrist_arr = data_grp.zeros(
            "wrist_image", shape=(0, out_h, out_w, 3), chunks=img_chunk, dtype="uint8", compressor=compressor
        )
    qpos_arr = data_grp.zeros("robot_qpos", shape=(0, 7), chunks=(1024, 7), dtype="float32", compressor=compressor)
    eef_arr = None
    if include_eef:
        eef_arr = data_grp.zeros("eef_pose", shape=(0, 6), chunks=(1024, 6), dtype="float32", compressor=compressor)
    action_arr = data_grp.zeros("action", shape=(0, 7), chunks=(1024, 7), dtype="float32", compressor=compressor)

    episode_ends: list[int] = []
    total_steps = 0
    skipped = []

    for ep_path in tqdm(hdf5_files, desc="Converting episodes"):
        with h5py.File(ep_path, "r") as ep:
            required = [
                "/observations/images/exterior_image_1_left",
                "/observations/qpos",
                "/action",
            ]
            if include_wrist:
                required.append("/observations/images/wrist_image_left")
            for key in required:
                if key not in ep:
                    raise KeyError(f"{ep_path.name}: missing key '{key}'")

            imgs_ext = ep["/observations/images/exterior_image_1_left"][:]  # (T,H,W,3)
            imgs_wrist = ep["/observations/images/wrist_image_left"][:] if include_wrist else None
            qpos = ep["/observations/qpos"][:].astype(np.float32)  # (T, 7)
            action = ep["/action"][:].astype(np.float32)  # (T, 7)
            if include_eef:
                if "/observations/eef_pose" not in ep:
                    raise KeyError(f"{ep_path.name}: missing eef_pose — use --no-eef for datasets recorded before EEF support was added")
                eef = ep["/observations/eef_pose"][:].astype(np.float32)
            else:
                eef = None

        T = imgs_ext.shape[0]
        if T == 0:
            skipped.append(ep_path.name)
            continue

        # ── Resize images if requested ────────────────────────────────────────
        if image_size:
            imgs_ext = np.stack([center_crop_resize(imgs_ext[t], image_size) for t in range(T)])
            if imgs_wrist is not None:
                imgs_wrist = np.stack([center_crop_resize(imgs_wrist[t], image_size) for t in range(T)])

        # ── Append to zarr arrays ─────────────────────────────────────────────
        ext_arr.append(imgs_ext)
        if wrist_arr is not None and imgs_wrist is not None:
            wrist_arr.append(imgs_wrist)
        qpos_arr.append(qpos)
        if eef_arr is not None and eef is not None:
            eef_arr.append(eef)
        action_arr.append(action)

        total_steps += T
        episode_ends.append(total_steps)

    meta_grp.array("episode_ends", np.array(episode_ends, dtype=np.int64), overwrite=True)

    if skipped:
        print(f"\nWarning: skipped {len(skipped)} empty episode(s): {skipped}")

    cams = "exterior + wrist" if include_wrist else "exterior only"
    print(f"\nDone — {len(episode_ends)} episodes, {total_steps} steps, {cams}")
    print(f"Zarr written to: {zarr_out}")
    print("\nZarr keys:")
    print(f"  data/exterior_image  {ext_arr.shape}  {ext_arr.dtype}")
    if wrist_arr is not None:
        print(f"  data/wrist_image     {wrist_arr.shape}  {wrist_arr.dtype}")
    print(f"  data/robot_qpos      {qpos_arr.shape}  {qpos_arr.dtype}")
    if eef_arr is not None:
        print(f"  data/eef_pose        {eef_arr.shape}  {eef_arr.dtype}")
    print(f"  data/action          {action_arr.shape}  {action_arr.dtype}")
    print(f"  meta/episode_ends    {len(episode_ends)} entries")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert UR5 HDF5 dataset to Zarr for Diffusion Policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, required=True, help="Directory containing episode_N.hdf5 files")
    p.add_argument("--zarr-out", type=Path, required=True, help="Output .zarr directory path")
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Resize images to this square size (e.g. 224). If omitted, stores at original recorded resolution.",
    )
    p.add_argument("--no-wrist", action="store_true", help="Exclude wrist camera from zarr output")
    p.add_argument("--no-eef", action="store_true", help="Exclude eef_pose from zarr output (for older datasets without it)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing zarr output")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert(
        raw_dir=args.raw_dir,
        zarr_out=args.zarr_out,
        image_size=args.image_size,
        include_wrist=not args.no_wrist,
        include_eef=not args.no_eef,
        overwrite=args.overwrite,
    )
