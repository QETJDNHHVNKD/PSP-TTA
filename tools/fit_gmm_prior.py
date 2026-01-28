import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from shape_loop import PolarFourierRenderer, ShapeClosedLoop

from utils.gmm_prior import GMMPrior, DiagGMMParams, OrgStats


from train import baseloader
def _try_import_scipy():
    try:
        import scipy.ndimage as ndi
        return ndi
    except Exception:
        return None


def preprocess_mask(bin_mask: np.ndarray) -> np.ndarray:

    ndi = _try_import_scipy()
    m = (bin_mask > 0).astype(np.bool_)

    if ndi is None:

        return m.astype(np.bool_)

    lab, n = ndi.label(m)
    if n <= 1:
        m2 = m
    else:
        areas = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep = 1 + int(np.argmax(areas))
        m2 = (lab == keep)

    m2 = ndi.binary_fill_holes(m2)
    return m2.astype(np.bool_)


def find_center(mask: np.ndarray) -> Tuple[int, int]:

    ndi = _try_import_scipy()
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2

    if ndi is not None:
        dist = ndi.distance_transform_edt(mask)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        return int(cy), int(cx)

    # fallback centroid
    return int(np.mean(ys)), int(np.mean(xs))


def radial_profile(mask: np.ndarray, cy: int, cx: int, N: int = 256) -> np.ndarray:

    ndi = _try_import_scipy()
    H, W = mask.shape

    if ndi is not None:
        er = ndi.binary_erosion(mask)
        bd = mask & (~er)
    else:
        bd = mask

    ys, xs = np.where(bd)
    if len(xs) == 0:
        ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((N,), dtype=np.float32)

    ys_lin = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xs_lin = np.linspace(-1.0, 1.0, W, dtype=np.float32)

    cx_n = xs_lin[cx]
    cy_n = ys_lin[cy]

    dx = xs_lin[xs] - cx_n
    dy = ys_lin[ys] - cy_n

    ang = np.arctan2(dy, dx)
    ang = (ang + 2 * np.pi) % (2 * np.pi)  # [0,2pi)
    rr = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    bins = np.floor(ang / (2 * np.pi) * N).astype(np.int64)
    bins = np.clip(bins, 0, N - 1)

    r = np.zeros((N,), dtype=np.float32)
    np.maximum.at(r, bins, rr)

    if (r == 0).any():
        idx = np.where(r > 0)[0]
        if len(idx) > 0:
            for i in range(N):
                if r[i] == 0:

                    d = np.minimum(np.abs(idx - i), N - np.abs(idx - i))
                    r[i] = r[idx[np.argmin(d)]]
    return r


def fit_fourier_r(r: np.ndarray, K: int, ridge: float = 1e-4) -> Tuple[float, np.ndarray, np.ndarray]:

    N = len(r)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False).astype(np.float32)

    cols = [np.ones((N, 1), dtype=np.float32)]
    for k in range(1, K + 1):
        cols.append(np.cos(k * theta)[:, None])
    for k in range(1, K + 1):
        cols.append(np.sin(k * theta)[:, None])

    A = np.concatenate(cols, axis=1)

    ATA = A.T @ A + ridge * np.eye(A.shape[1], dtype=np.float32)
    ATy = A.T @ r.astype(np.float32)
    w = np.linalg.solve(ATA, ATy)  # [1+2K]

    r0 = float(w[0])
    a = w[1:1 + K].astype(np.float32)
    b = w[1 + K:].astype(np.float32)
    return r0, a, b


def mask_to_zshape(mask_bin: np.ndarray, K: int, N: int, r_max: float) -> Tuple[np.ndarray, np.ndarray]:

    H, W = mask_bin.shape
    m = preprocess_mask(mask_bin)
    cy, cx = find_center(m)

    ys_lin = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xs_lin = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    cx_n = xs_lin[cx]
    cy_n = ys_lin[cy]

    r = radial_profile(m, cy, cx, N=N)  # [N]
    r0, a, b = fit_fourier_r(r, K=K, ridge=1e-4)


    a_lim = 0.15 * r_max
    b_lim = 0.15 * r_max

    r0 = np.clip(r0, 1e-3, r_max).astype(np.float32)
    a = np.clip(a, -a_lim, a_lim)
    b = np.clip(b, -b_lim, b_lim)

    z_shape = np.concatenate([[r0], a, b], axis=0).astype(np.float32)
    center = np.array([cx_n, cy_n], dtype=np.float32)
    return z_shape, center


def fit_gmm_diag(Z: np.ndarray, k_candidates=(1, 2, 4, 8, 12), reg_covar=1e-5, seed=0):
    from sklearn.mixture import GaussianMixture

    best = None
    best_bic = float("inf")
    for k in k_candidates:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=reg_covar,
            n_init=10,
            max_iter=500,
            random_state=seed,
            init_params="kmeans",
        )
        gmm.fit(Z)
        bic = gmm.bic(Z)
        if bic < best_bic:
            best_bic = bic
            best = gmm
    return best, best_bic


def _pca2(Z: np.ndarray):
    Zm = Z.mean(axis=0, keepdims=True)
    Z0 = Z - Zm
    U, S, Vt = np.linalg.svd(Z0, full_matrices=False)
    W = Vt[:2].T  # [D,2]
    Z2 = Z0 @ W   # [N,2]
    return Z2, W, Zm

def _visualize_prior(prior, Z_by_org, device="cpu"):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    device = torch.device(device) if not isinstance(device, torch.device) else device
    prior = prior.to(device)

    prior.eval()
    org_ids = prior.org_ids


    for oid in org_ids:
        w = getattr(prior, f"w_{oid}").detach().cpu().numpy()
        mu = getattr(prior, f"mu_{oid}").detach().cpu().numpy()

        var_t = getattr(prior, f"var_{oid}").detach().clamp_min(1e-4)
        var = var_t.cpu().numpy()

        print(f"[org={oid}] w: {w.shape}, mu: {mu.shape}, var: {var.shape}, "
              f"min(var)={var.min():.3e}, max(var)={var.max():.3e}, sum(w)={w.sum():.4f}")


   
    for oid in org_ids:
        Z = np.stack(Z_by_org[oid], axis=0)
        z = torch.from_numpy(Z).to(device).float()

        nll_c = prior.nll(z, organ_id=int(oid), reduction="none").detach().cpu().numpy()
        nll_m = prior.nll(z, organ_id=None, reduction="none").detach().cpu().numpy()


        lps = []
        for oid2 in org_ids:
            lp = prior.log_prob_conditional(z, int(oid2))  
            lps.append(lp)
        lps = torch.stack(lps, dim=1)  # [B,O]
        pred_idx = lps.argmax(dim=1).detach().cpu().numpy()
        target_idx = org_ids.index(oid)
        acc = (pred_idx == target_idx).mean()

        print(f"[org={oid}] N={len(Z)} | "
              f"condNLL mean={nll_c.mean():.3f} std={nll_c.std():.3f} | "
              f"mixNLL mean={nll_m.mean():.3f} std={nll_m.std():.3f} | "
              f"argmax-org-acc={acc:.3f}")

    
        plt.figure()
        plt.hist(nll_c, bins=50, alpha=0.7, label=f"cond NLL (org={oid})")
        plt.hist(nll_m, bins=50, alpha=0.7, label="US-mix NLL")
        plt.title(f"NLL Histogram (org={oid})")
        plt.xlabel("NLL")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()


  
    all_Z = []
    all_y = []
    for oid in org_ids:
        Z = np.stack(Z_by_org[oid], axis=0)
        all_Z.append(Z)
        all_y.append(np.full((Z.shape[0],), oid))
    all_Z = np.concatenate(all_Z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    Z2, W, Zm = _pca2(all_Z)

    plt.figure()
    for oid in org_ids:
        idx = (all_y == oid)
        plt.scatter(Z2[idx, 0], Z2[idx, 1], s=6, label=f"org={oid}")
    plt.title("z_shape PCA-2D (dataset samples)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for oid in org_ids:
        mu_std = getattr(prior, f"mu_{oid}").detach().cpu().numpy()
        mean = getattr(prior, f"mean_{oid}").detach().cpu().numpy()   
        std = getattr(prior, f"std_{oid}").detach().cpu().numpy()     
        mu_phys = mu_std * std[None, :] + mean[None, :]

        mu2 = (mu_phys - Zm) @ W    # [M,2]
        plt.scatter(mu2[:, 0], mu2[:, 1], s=40, label=f"org={oid} means")
    plt.title("GMM component means in PCA-2D (physical space)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data_configuration", required=True)
    parser.add_argument("--input_size", default=224, type=int)

    parser.add_argument("--K", default=8, type=int)
    parser.add_argument("--N", default=256, type=int)
    parser.add_argument("--out", default="gmm_prior_us.pth")
    parser.add_argument("--seed", default=0, type=int)


    parser.add_argument("--derm_id", default=3, type=int)
    parser.add_argument("--org_ids", default="0,1,2")  # US organs

    parser.add_argument("--stage1_ckpt", default=None, type=str)
    parser.add_argument("--mode", default="geom", choices=["geom", "encode"])
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis_only", action="store_true")
    parser.add_argument("--prior_path", default=None, type=str)
    parser.add_argument("--vis_max_per_org", default=800, type=int)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    train_loader, val_loader, _ = baseloader(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    shape_loop = None
    if args.stage1_ckpt is not None:

        shape_loop = ShapeClosedLoop(out_hw=args.input_size, K=args.K, sharpness=40.0).to(device)
        ckpt = torch.load(args.stage1_ckpt, map_location=device)
        state = ckpt["shape_loop"] if (isinstance(ckpt, dict) and "shape_loop" in ckpt) else ckpt
        shape_loop.load_state_dict(state, strict=True)
        shape_loop.eval()
        for p in shape_loop.parameters():
            p.requires_grad_(False)

        renderer = shape_loop.renderer
    else:
        renderer = PolarFourierRenderer(args.input_size, args.input_size, K=args.K, sharpness=40.0, r_max=1.25)

    r_max = float(renderer.r_max)


    org_ids = [int(x) for x in args.org_ids.split(",")]

    Z_by_org: Dict[int, List[np.ndarray]] = {oid: [] for oid in org_ids}

    def _consume(loader):
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                IMG = batch[0]
                MSK = batch[1]
                setseq = batch[3] if len(batch) >= 4 else batch[2]

            else:
                continue

            MSK = MSK.cpu()
            setseq = setseq.cpu().long()

            
            for i in range(MSK.shape[0]):
                oid = int(setseq[i].item())
                if oid == args.derm_id:
                    continue
                if oid not in Z_by_org:
                    continue


                m = MSK[i].numpy()
                if m.ndim == 3:
                    m = m[0]  # [H,W]
                bin_mask = np.isin(m, [1, 2]).astype(np.uint8)

                if bin_mask.sum() < 10:
                    continue

                if args.mode == "geom":
                    z_shape, _center = mask_to_zshape(bin_mask, K=args.K, N=args.N, r_max=r_max)
                else:

                    gt = torch.from_numpy(bin_mask.astype(np.float32))[None, None].to(device) 

                    if gt.shape[-2:] != (args.input_size, args.input_size):
                        gt = F.interpolate(gt, size=(args.input_size, args.input_size), mode="nearest")

                    with torch.no_grad():
                        z_raw = shape_loop.mask_encoder(gt)

                        _, _, _, r0, a, b = renderer.decode(z_raw)
                        z_shape = torch.cat([r0, a, b], dim=1).squeeze(0).cpu().numpy().astype(np.float32)

                Z_by_org[oid].append(z_shape)


    _consume(train_loader)
    _consume(val_loader)

    org_gmms = {}
    org_stats = {}
    org_mix = {}

    for oid in org_ids:
        Z = np.stack(Z_by_org[oid], axis=0)  # [n,D]
        print(f"[org={oid}] collected {Z.shape[0]} samples, D={Z.shape[1]}")
        if Z.shape[0] < 30:
            raise RuntimeError(f"Too few samples for organ {oid}: {Z.shape[0]}")

        mean = Z.mean(axis=0, keepdims=True)
        std = Z.std(axis=0, keepdims=True) + 1e-6
        Zs = (Z - mean) / std

        if args.vis_only:
            prior_path = args.prior_path or args.out
            prior = GMMPrior.load(prior_path, device=device)
            _visualize_prior(prior, Z_by_org, device=device)
            return

        gmm, bic = fit_gmm_diag(
            Zs,
            k_candidates=(1, 2, 4, 8),
            reg_covar=1e-3,
            seed=args.seed
        )

        org_stats[oid] = OrgStats(
            mean=torch.from_numpy(mean.squeeze(0)).float(),
            std=torch.from_numpy(std.squeeze(0)).float(),
        )
        org_gmms[oid] = DiagGMMParams(
            weights=torch.from_numpy(gmm.weights_).float(),
            means=torch.from_numpy(gmm.means_).float(),
            vars=torch.from_numpy(gmm.covariances_).float(),
        )
        org_mix[oid] = float(Z.shape[0])


    s = sum(org_mix.values())
    org_mix = {k: v / s for k, v in org_mix.items()}

    prior = GMMPrior(org_gmms=org_gmms, org_stats=org_stats, org_mix_weights=org_mix)
    prior.save(args.out)

    if args.vis or args.vis_only:
        _visualize_prior(prior, Z_by_org, device=device)


if __name__ == "__main__":
    main()
