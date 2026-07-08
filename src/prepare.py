"""Dataset preparation: filter to ball-only labels, combine train/test splits
from the raw Soccana layout into one pool, subsample, and re-split into a
fresh train/val set for YOLO training.

The raw dataset is nested by split (images/train, images/test, labels/train,
labels/test). Soccana's own train/test split isn't a meaningful held-out set
(same source videos appear in both), so we combine both splits into one pool
before doing our own subsample + 80/20 split. Some filenames collide between
train/ and test/ (different content, same name), so the combining step
prefixes each stem by its source split to guarantee no collision.
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml


def filter_ball_class(labels_dir: Path, ball_class_id: int) -> list[str]:
    """Rewrite each label file in labels_dir to keep only ball-class lines
    (remapped to class 0). Returns stems of files that still have >=1 line."""
    kept = []
    for lbl_file in sorted(labels_dir.glob("*.txt")):
        lines = lbl_file.read_text().splitlines()
        ball_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) == ball_class_id:
                ball_lines.append("0 " + " ".join(parts[1:]))
        if ball_lines:
            lbl_file.write_text("\n".join(ball_lines) + "\n")
            kept.append(lbl_file.stem)
    return kept


def discover_splits(input_dir: Path) -> list[str]:
    """Return sorted split names present under both images/ and labels/.

    If neither has split subdirectories (flat layout), return [""] as a
    sentinel meaning "use input_dir/images and input_dir/labels directly."
    """
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    splits = []
    if images_dir.is_dir() and labels_dir.is_dir():
        for entry in images_dir.iterdir():
            if entry.is_dir() and (labels_dir / entry.name).is_dir():
                splits.append(entry.name)

    if not splits:
        return [""]
    return sorted(splits)


def combine_and_filter(
    input_dir: Path, ball_class_id: int
) -> dict[str, tuple[Path, Path, str]]:
    """Filter each discovered split to ball-only labels and merge the
    results into one dict keyed by split-prefixed stem, so filenames that
    collide across splits never overwrite one another."""
    combined: dict[str, tuple[Path, Path, str]] = {}
    for split in discover_splits(input_dir):
        if split == "":
            img_dir = input_dir / "images"
            lbl_dir = input_dir / "labels"
        else:
            img_dir = input_dir / "images" / split
            lbl_dir = input_dir / "labels" / split

        kept_stems = filter_ball_class(lbl_dir, ball_class_id)
        for orig_stem in kept_stems:
            combined_key = f"{split}_{orig_stem}" if split != "" else orig_stem
            combined[combined_key] = (img_dir, lbl_dir, orig_stem)

    return combined


def subsample(stems: list[str], n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    if n >= len(stems):
        return list(stems)
    return rng.sample(stems, n)


def split_dataset(
    stems: list[str], val_frac: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_frac))
    return shuffled[:split], shuffled[split:]


def write_data_yaml(output_dir: Path, dataset_path: str) -> None:
    config = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["ball"],
    }
    (output_dir / "data.yaml").write_text(yaml.dump(config, default_flow_style=False))


def copy_split(
    stems: list[str],
    source_map: dict[str, tuple[Path, Path, str]],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        img_dir, lbl_dir, orig_stem = source_map[stem]
        for ext in (".jpg", ".jpeg", ".png"):
            src_img = img_dir / f"{orig_stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, dst_img_dir / f"{stem}{src_img.suffix}")
                break
        src_lbl = lbl_dir / f"{orig_stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl_dir / f"{stem}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a ball-only YOLO dataset from the raw Soccana dataset."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ball-class-id", required=True, type=int)
    parser.add_argument("--n", default=5000, type=int)
    parser.add_argument("--val-frac", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--drive-path",
        default="/content/drive/MyDrive/SoccerDetection/data/processed",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = discover_splits(args.input)
    source_map = combine_and_filter(args.input, args.ball_class_id)

    selected = subsample(list(source_map.keys()), n=args.n, seed=args.seed)
    train_stems, val_stems = split_dataset(
        selected, val_frac=args.val_frac, seed=args.seed
    )

    copy_split(
        train_stems, source_map, output_dir / "images" / "train", output_dir / "labels" / "train"
    )
    copy_split(
        val_stems, source_map, output_dir / "images" / "val", output_dir / "labels" / "val"
    )

    write_data_yaml(output_dir, dataset_path=args.drive_path)

    print(f"Discovered {len(splits)} split(s): {splits}")
    print(f"Images surviving filtering (combined pool): {len(source_map)}")
    print(f"Selected for dataset: {len(selected)}")
    print(f"Train: {len(train_stems)} | Val: {len(val_stems)}")


if __name__ == "__main__":
    main()
