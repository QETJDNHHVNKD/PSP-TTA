#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_ids(domain_root: Path):
    img_dir = domain_root / "images"
    mask_dir = domain_root / "masks"
    images = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    masks = [p for p in mask_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    mask_stems = {p.stem for p in masks}
    ids = sorted(p.stem for p in images if p.stem in mask_stems)
    if not ids:
        raise RuntimeError(f"No matched image-mask basenames found in {domain_root}")
    return ids

def sha256_lines(lines):
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()

def split_ids(ids, seed, val_fraction):
    items = list(ids)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_fraction)))
    return sorted(items[n_val:]), sorted(items[:n_val])

def write_txt(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("splits/generated"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--val-fraction", type=float, default=0.20)
    args = ap.parse_args()

    domains = ["BUSI", "TN3K"]
    all_ids = {d: list_ids(args.data_root / d) for d in domains}

    for seed in args.seeds:
        meta = {"seed": seed, "val_fraction": args.val_fraction, "domains": {}}
        combined_train, combined_val = [], []

        for domain in domains:
            train, val = split_ids(all_ids[domain], seed, args.val_fraction)
            write_txt(args.out_dir / f"seed_{seed}" / f"{domain}_train.txt", train)
            write_txt(args.out_dir / f"seed_{seed}" / f"{domain}_val.txt", val)
            combined_train += [f"{domain}/{x}" for x in train]
            combined_val += [f"{domain}/{x}" for x in val]
            meta["domains"][domain] = {
                "total": len(all_ids[domain]),
                "train": len(train),
                "val": len(val),
                "all_ids_sha256": sha256_lines(all_ids[domain]),
                "train_ids_sha256": sha256_lines(train),
                "val_ids_sha256": sha256_lines(val),
            }

        write_txt(args.out_dir / f"seed_{seed}" / "public_train.txt", sorted(combined_train))
        write_txt(args.out_dir / f"seed_{seed}" / "public_val.txt", sorted(combined_val))
        (args.out_dir / f"seed_{seed}" / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"seed={seed}: train={len(combined_train)}, val={len(combined_val)}")

if __name__ == "__main__":
    main()
