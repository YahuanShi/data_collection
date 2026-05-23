#!/usr/bin/env python3
"""
Resize / Crop Images in HDF5 Episodes
======================================

Converts all camera image streams in HDF5 episodes from the recorded resolution
(e.g. 480x480) to the target training resolution (default 224x224).

Intended as the final step in the data-processing pipeline, after trimming and
smoothing, producing a training-ready dataset. Compatible with openpi / pi0.5
training as well as other robot-learning frameworks (LeRobot, ACT, Diffusion
Policy, etc.) that expect fixed-size image inputs.

Three preprocessing strategies are available via --method:

    center_crop   Square-crop the centre of the image, then resize to target.
                  Preserves aspect ratio; discards border content.
                  Best when the task-relevant region is always centred.

    resize        Resize the full image directly to target (may squish if not
                  already square). Preserves all content at lower resolution.
                  Best when global context matters across the entire frame.

    pad_resize    Resize while preserving aspect ratio, padding remainder with
                  black. No squishing; background padding added as needed.

Datasets produced by the episode recorder are already square (center-crop +
resize applied at capture time), so center_crop and resize are equivalent for
perfectly square source images — choose based on your training setup.

Usage:
    python3 05_resize_images.py src [dst]
    python3 05_resize_images.py path/to/trimmed_dir        processed_data/resized/
    python3 05_resize_images.py path/to/episode_0.hdf5     episode_0_224.hdf5
    python3 05_resize_images.py path/to/trimmed_dir        processed_data/resized/ --size 256 --method resize
    python3 05_resize_images.py path/to/trimmed_dir        --dry-run

Arguments:
    src         Single .hdf5 file or directory containing *.hdf5 files
    dst         Output file (single-file mode) or directory (batch mode).
                Omit to use --dry-run only.
    --size      Target square size in pixels (default: 224)
    --method    Preprocessing strategy: center_crop | resize | pad_resize
                (default: center_crop)
    --dry-run   Print what would be done without writing any files
    --workers   Number of parallel worker processes (default: 4)

HDF5 layout expected (openpi / ALOHA compatible):
    /observations/images/exterior_image_1_left   (T, H, W, 3) uint8  RGB
    /observations/images/wrist_image_left         (T, H, W, 3) uint8  RGB
    /observations/images/front_image_1            (T, H, W, 3) uint8  RGB  [optional]
    /observations/qpos                            (T, D) float64
    /action                                       (T, D) float64
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import cv2
import h5py
import numpy as np

# ── Image transform functions ──────────────────────────────────────────────────


def _center_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Square-crop the centre, then resize to size x size."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize the full image directly to size x size."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _pad_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad remainder with black."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


_METHODS = {
    "center_crop": _center_crop_resize,
    "resize": _resize,
    "pad_resize": _pad_resize,
}


# ── Per-episode processing ─────────────────────────────────────────────────────


def _image_keys(f: h5py.File) -> list[str]:
    grp = f.get("observations/images")
    if grp is None:
        return []
    return list(grp.keys())


def process_episode(src_path: str, dst_path: str, size: int, method: str) -> str:
    """Read one HDF5 episode, resize all image streams, write to dst_path.

    Returns a short status string.
    """
    transform = _METHODS[method]

    with h5py.File(src_path, "r") as src:
        qpos = src["observations/qpos"][:]
        action = src["action"][:]
        qvel = src["observations/qvel"][:] if "observations/qvel" in src else None
        eef_pose = src["observations/eef_pose"][:] if "observations/eef_pose" in src else None
        attrs = dict(src.attrs)
        img_keys = _image_keys(src)

        # Load and transform all image streams
        images: dict[str, np.ndarray] = {}
        for key in img_keys:
            raw = src[f"observations/images/{key}"][:]  # (T, H, W, 3)
            images[key] = np.stack([transform(frame, size) for frame in raw])

    with h5py.File(dst_path, "w") as dst:
        for k, v in attrs.items():
            dst.attrs[k] = v
        # Update image-size metadata if present
        dst.attrs["image_size"] = size

        dst.create_dataset("observations/qpos", data=qpos, compression="gzip")
        dst.create_dataset("action", data=action, compression="gzip")
        if qvel is not None:
            dst.create_dataset("observations/qvel", data=qvel, compression="gzip")
        if eef_pose is not None:
            dst.create_dataset("observations/eef_pose", data=eef_pose, compression="gzip")
        for key, arr in images.items():
            dst.create_dataset(
                f"observations/images/{key}",
                data=arr,
                compression="gzip",
                chunks=(1, size, size, 3),  # one frame per chunk — fast random access
            )

    src_h = next(iter(images.values())).shape[1] if images else "?"
    return f"{os.path.basename(src_path)}  {src_h}px → {size}px  [{','.join(img_keys)}]"


def _worker(args):
    src_path, dst_path, size, method = args
    try:
        msg = process_episode(src_path, dst_path, size, method)
        return True, msg
    except Exception as exc:
        return False, f"ERROR {os.path.basename(src_path)}: {exc}"


# ── Main ───────────────────────────────────────────────────────────────────────


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
    parser = argparse.ArgumentParser(
        description="Resize all camera images in HDF5 episodes to a target training resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("src", help="Single .hdf5 file or directory containing *.hdf5 files")
    parser.add_argument("dst", nargs="?", default=None, help="Output file or directory (omit for --dry-run)")
    parser.add_argument("--size", type=int, default=224, help="Target square image size in pixels")
    parser.add_argument(
        "--method",
        choices=list(_METHODS),
        default="center_crop",
        help="Image preprocessing strategy",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing files")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    args = parser.parse_args()

    if not args.dry_run and args.dst is None:
        sys.exit("ERROR: dst is required unless --dry-run is set")

    files = collect_files(args.src)
    single = os.path.isfile(args.src)

    print(f"\n{'─' * 60}")
    print(f"  Source  : {args.src}  ({len(files)} file{'s' if len(files) != 1 else ''})")
    print(f"  Dest    : {args.dst or '(dry-run)'}")
    print(f"  Size    : {args.size}x{args.size}")
    print(f"  Method  : {args.method}")
    if not single:
        print(f"  Workers : {args.workers}")
    if args.dry_run:
        print("  Mode    : DRY RUN — no files will be written")
    print(f"{'─' * 60}\n")

    if args.dry_run:
        for f in files:
            print(f"  would process: {os.path.basename(f)}")
        print(f"\nDry run complete — {len(files)} episode(s) would be processed.")
        return

    if single:
        ok, msg = _worker((files[0], args.dst, args.size, args.method))
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            sys.exit(1)
    else:
        os.makedirs(args.dst, exist_ok=True)
        work = [(path, os.path.join(args.dst, os.path.basename(path)), args.size, args.method) for path in files]
        ok_n = err_n = 0
        with mp.Pool(processes=args.workers) as pool:
            for success, msg in pool.imap_unordered(_worker, work):
                if success:
                    ok_n += 1
                    print(f"  OK   {msg}")
                else:
                    err_n += 1
                    print(f"  FAIL {msg}", file=sys.stderr)

        print(f"\n{'─' * 60}")
        print(f"  Done — {ok_n} OK, {err_n} failed  →  {args.dst}")
    print(f"{'─' * 60}\n")
    if err_n:
        sys.exit(1)


if __name__ == "__main__":
    main()
