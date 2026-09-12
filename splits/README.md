# Public-only split manifests

Generate deterministic BUSI/TN3K train/validation manifests with:

```bash
python scripts/build_public_splits.py --data-root /path/to/data \
  --out-dir splits/generated --seeds 2024 2025 2026 --val-fraction 0.20
```

Expected data layout:
DATA_ROOT/BUSI/images, DATA_ROOT/BUSI/masks
DATA_ROOT/TN3K/images, DATA_ROOT/TN3K/masks


