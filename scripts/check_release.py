#!/usr/bin/env python3
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT/"configs"/"public_only.yaml",
    ROOT/"scripts"/"build_public_splits.py",
    ROOT/"scripts"/"fit_source_prior.py",
    ROOT/"splits"/"README.md",
    ROOT/"prior_stats"/"source_calibration.json",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing:\n  " + "\n  ".join(missing))

cal = json.loads((ROOT/"prior_stats"/"source_calibration.json").read_text(encoding="utf-8"))
if cal.get("q5_src") is None or cal.get("q95_src") is None:
    print("WARNING: q5_src/q95_src are not populated. Run fit_source_prior.py before claiming exact end-to-end reproduction.")
else:
    assert cal["q95_src"] > cal["q5_src"], "q95_src must exceed q5_src"
    print("Calibration constants valid.")
print("Release structure OK.")
