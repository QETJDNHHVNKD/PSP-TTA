#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture

EPS = 1e-8

def load_codes(path: Path):
    d = np.load(path)
    train = np.asarray(d["train"], dtype=np.float64)
    val = np.asarray(d["val"], dtype=np.float64)
    if train.ndim != 2 or val.ndim != 2 or train.shape[1] != 17 or val.shape[1] != 17:
        raise ValueError("Expected train/val arrays with shape [N,17].")
    return train, val

def fit_domain(train, n_components, seed):
    mu = train.mean(0)
    sigma = np.maximum(train.std(0), EPS)
    z = (train - mu) / sigma
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=seed,
        reg_covar=1e-6,
        max_iter=500,
        n_init=5,
    ).fit(z)
    return mu, sigma, gmm

def log_density_physical(x, mu, sigma, gmm):
    z = (x - mu) / sigma
    return gmm.score_samples(z) - np.log(sigma).sum()

def logsumexp(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)

def export_gmm(gmm):
    return {
        "weights": gmm.weights_.tolist(),
        "means_standardized": gmm.means_.tolist(),
        "covariances_diag_standardized": gmm.covariances_.tolist(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi", type=Path, required=True)
    ap.add_argument("--tn3k", type=Path, required=True)
    ap.add_argument("--components", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--mixing", choices=["proportional", "uniform"], default="proportional")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = {}
    for name, path in [("BUSI", args.busi), ("TN3K", args.tn3k)]:
        train, val = load_codes(path)
        mu, sigma, gmm = fit_domain(train, args.components, args.seed)
        data[name] = dict(train=train, val=val, mu=mu, sigma=sigma, gmm=gmm)

    if args.mixing == "proportional":
        n = np.array([len(data["BUSI"]["train"]), len(data["TN3K"]["train"])], float)
        alpha = n / n.sum()
    else:
        alpha = np.array([0.5, 0.5])

    val_pool = np.concatenate([data["BUSI"]["val"], data["TN3K"]["val"]], axis=0)
    logps = []
    for w, name in zip(alpha, ["BUSI", "TN3K"]):
        d = data[name]
        logps.append(np.log(w + EPS) + log_density_physical(val_pool, d["mu"], d["sigma"], d["gmm"]))
    nll = -logsumexp(np.vstack(logps), axis=0)

    q5 = float(np.percentile(nll, 5))
    q95 = float(np.percentile(nll, 95))

    out = {
        "schema_version": 1,
        "source_domains": ["BUSI", "TN3K"],
        "shape_code_order": ["r0"] + [f"a{k}" for k in range(1,9)] + [f"b{k}" for k in range(1,9)],
        "gmm_components_per_source": args.components,
        "covariance_type": "diag",
        "mixing_rule": args.mixing,
        "alpha": {"BUSI": float(alpha[0]), "TN3K": float(alpha[1])},
        "q5_src": q5,
        "q95_src": q95,
        "domains": {}
    }

    for name in ["BUSI", "TN3K"]:
        d = data[name]
        out["domains"][name] = {
            "n_train_codes": int(len(d["train"])),
            "n_val_codes": int(len(d["val"])),
            "standardization_mean": d["mu"].tolist(),
            "standardization_std": d["sigma"].tolist(),
            "gmm": export_gmm(d["gmm"]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"q5_src={q5:.8f}")
    print(f"q95_src={q95:.8f}")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
