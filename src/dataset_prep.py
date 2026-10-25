"""
Download the Kaggle fish-disease datasets, sort them into dataset/raw/<class>/,
then make a stratified 70/15/15 train/val/test split under dataset/split/.

Run once before training (safe to re-run after editing CLASS_KEYWORDS -- it
merges into existing dataset/raw/ rather than wiping it):
    python src/dataset_prep.py

Requires a Kaggle account + API token (https://www.kaggle.com/settings ->
API section -> "Create Legacy API Key" -> save kaggle.json to
~/.kaggle/kaggle.json, or set the KAGGLE_USERNAME / KAGGLE_KEY environment
variables). kagglehub picks these up automatically.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import kagglehub

from config import CLASS_KEYWORDS, CLASS_NAMES, RAW_DATA_DIR, SEED, SPLIT_DATA_DIR, \
    TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT

KAGGLE_DATASETS = [
    "subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia",
    "utpolkantidas/fresh-water-fish-disease-dataset",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def match_class(folder_name: str) -> str | None:
    name = folder_name.lower()
    for class_name, keywords in CLASS_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return class_name
    return None


def collect_source_images(download_root: Path, buckets: dict[str, list[Path]],
                           unmatched_dirs: set[str]) -> None:
    """Walk one downloaded dataset and bucket its image files by matched class."""
    for path in download_root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        folder_name = path.parent.name
        matched = match_class(folder_name)
        if matched:
            buckets[matched].append(path)
        else:
            unmatched_dirs.add(folder_name)


def copy_into_raw(buckets: dict[str, list[Path]]) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for class_name, files in buckets.items():
        class_dir = RAW_DATA_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(files):
            dst = class_dir / f"{class_name}_{i:04d}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copyfile(src, dst)
        print(f"  {class_name:30s} {len(files):4d} images")


def stratified_split() -> None:
    random.seed(SEED)
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            (SPLIT_DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    print("\nSplitting 70/15/15 (train/val/test):")
    for class_name in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / class_name
        files = sorted(class_dir.glob("*"))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, split_files in splits.items():
            for src in split_files:
                dst = SPLIT_DATA_DIR / split / class_name / src.name
                shutil.copyfile(src, dst)
        print(f"  {class_name:30s} train={len(splits['train']):4d} "
              f"val={len(splits['val']):4d} test={len(splits['test']):4d}")


def main() -> None:
    buckets: dict[str, list[Path]] = {c: [] for c in CLASS_NAMES}
    unmatched_dirs: set[str] = set()

    for dataset_slug in KAGGLE_DATASETS:
        print(f"Downloading '{dataset_slug}' from Kaggle (cached after first run)...")
        download_root = Path(kagglehub.dataset_download(dataset_slug))
        print(f"Downloaded to: {download_root}")
        collect_source_images(download_root, buckets, unmatched_dirs)

    if unmatched_dirs:
        print(f"\n[warn] {len(unmatched_dirs)} folder(s) didn't match any class "
              f"and were skipped: {sorted(unmatched_dirs)}")
        print("       Extend CLASS_KEYWORDS in src/config.py if one of these "
              "should map to a class.")

    print("\nSorting images into dataset/raw/<class>/:")
    empty = [c for c, files in buckets.items() if not files]
    if empty:
        print(f"[warn] no images found for: {empty} "
              "(check CLASS_KEYWORDS match the dataset's folder names)")
    copy_into_raw(buckets)

    stratified_split()
    print(f"\nDone. Split dataset ready at: {SPLIT_DATA_DIR}")


if __name__ == "__main__":
    main()
