import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TTACfg:
    steps: int = 5
    lr: float = 5e-2

    # pseudo-label selection
    thr_hi: float = 0.95          # start threshold
    thr_lo: float = 0.90          # end threshold (relax boundary)
    max_ratio: float = 0.20       # keep at most top-ratio pixels
    w_pl: float = 0.2             # weight for pseudo-label loss

    # regularizers
    beta_cycle: float = 1.0       # cycle strength
    lambda0: float = 1.0          # base prior weight (will be gated by confidence)
    gamma0: float = 0.2           # base trust/anchor weight (will be gated by confidence)
    w_area: float = 0.05          # keep area close to initial (small, but helps anti-shrink)
    nll_scale: float = 1.0        # same meaning as train_stage2.py

    # drift protection (raw z clamp)
    clip_p: float = 6.0
    clip_c: float = 4.0
    clip_r: float = 6.0
    clip_ab: float = 4.0

    # stop/skip protection
    min_selected_ratio: float = 1e-4  # if almost no confident pixels, stop early


def _topk_cap(w: torch.Tensor, max_ratio: float) -> torch.Tensor:
    """Keep only top max_ratio entries per-sample (w in [0,1])."""
    if max_ratio >= 1.0:
        return w
    B = w.shape[0]
    flat = w.view(B, -1)
    N = flat.shape[1]
    k = int(max(1, min(N, round(max_ratio * N))))
    _, topi = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
    sel = torch.zeros_like(flat)
    sel.scatter_(1, topi, 1.0)
    return (flat * sel).view_as(w)


def _weighted_pl_loss(prob: torch.Tensor, thr: float, max_ratio: float):
    """
    prob: [B,1,H,W] in (0,1)
    returns: lpl, stats
    """
    prob = prob.clamp(1e-6, 1 - 1e-6)
    conf = torch.maximum(prob, 1.0 - prob)  # [B,1,H,W]

    # soft weight 0..1 based on confidence
    w0 = ((conf - thr) / (1.0 - thr)).clamp(0.0, 1.0)
    w0 = _topk_cap(w0, max_ratio=max_ratio)

    # pseudo label from current prediction
    y = (prob > 0.5).float()

    # fg/bg balancing (DERM 背景极多，这步很关键)
    w_pos = w0 * y
    w_neg = w0 * (1.0 - y)
    sum_pos = w_pos.sum((1, 2, 3)).clamp_min(1.0)
    sum_neg = w_neg.sum((1, 2, 3)).clamp_min(1.0)
    scale = (sum_neg / sum_pos).clamp(1.0, 10.0).view(-1, 1, 1, 1)
    w = w_pos * scale + w_neg

    wsum = w.sum((1, 2, 3))  # [B]
    has = (wsum > 0).float()
    if has.sum() < 0.5:
        z = torch.tensor(0.0, device=prob.device)
        ratio = ((w0 > 0).float().sum((1, 2, 3)) / float(w0[0].numel())).mean().detach()
        wpl = torch.tensor(0.0, device=prob.device)
        return z, {"pl_ratio": ratio, "wpl": wpl, "has": has.mean().detach()}

    bce_map = F.binary_cross_entropy(prob, y, reduction="none")
    bce = (bce_map * w).sum((1, 2, 3)) / wsum.clamp_min(1.0)
    bce = (bce * has).sum() / has.sum().clamp_min(1.0)

    inter = ((prob * y) * w).sum((1, 2, 3)) * 2.0
    denom = ((prob * w).sum((1, 2, 3)) + (y * w).sum((1, 2, 3)) + 1e-6)
    dice = (inter / denom)
    dice = (dice * has).sum() / has.sum().clamp_min(1.0)
    ldice = 1.0 - dice

    lpl = bce + ldice

    ratio = ((w0 > 0).float().sum((1, 2, 3)) / float(w0[0].numel())).mean().detach()
    sel = (w0 > 0).float()
    wpl = (conf * sel).sum((1, 2, 3)) / sel.sum((1, 2, 3)).clamp_min(1.0)
    wpl = wpl.mean().detach()
    return lpl, {"pl_ratio": ratio, "wpl": wpl, "bce": bce.detach(), "ldice": ldice.detach()}


def _clamp_z_raw_(z: torch.Tensor, cfg: TTACfg):
    # z layout: [p, cx, cy, r0, a1..aK, b1..bK]
    z[:, 0].clamp_(-cfg.clip_p, cfg.clip_p)
    z[:, 1:3].clamp_(-cfg.clip_c, cfg.clip_c)
    z[:, 3].clamp_(-cfg.clip_r, cfg.clip_r)
    z[:, 4:].clamp_(-cfg.clip_ab, cfg.clip_ab)


@torch.no_grad()
def forward_z_m(model, shape_loop, img: torch.Tensor):
    feats = model.forward_features(img)
    z1, m1, _, _ = shape_loop(feats)
    return z1, m1


def tta_z_only(
    model,
    shape_loop,
    img: torch.Tensor,
    prior=None,
    extract_z_shape_fn=None,
    cfg: TTACfg = TTACfg(),
):
    """
    Test-time adaptation for derm: update ONLY z (no model weights).

    Returns:
      m_best: [B,1,H,W]
      logs: dict of scalars
    """
    device = img.device
    model.eval()
    shape_loop.eval()

    # ---- init ----
    with torch.no_grad():
        z0, m0 = forward_z_m(model, shape_loop, img)

    z = z0.detach().clone().requires_grad_(True)
    area0 = m0.detach().mean(dim=(1, 2, 3), keepdim=False)  # [B]

    opt = torch.optim.Adam([z], lr=cfg.lr)

    best = {"J": float("inf"), "m": m0.detach(), "step": -1}
    logs: Dict[str, float] = {}

    for k in range(cfg.steps):
        thr = cfg.thr_hi + (cfg.thr_lo - cfg.thr_hi) * (k / max(1, cfg.steps - 1))

        opt.zero_grad(set_to_none=True)

        # 当前预测
        m = shape_loop.renderer(z)  # [B,1,H,W], sigmoid 后

        # 1) PL loss (高置信像素)
        lpl, st = _weighted_pl_loss(m, thr=thr, max_ratio=cfg.max_ratio)

        # 2) cycle (防 z 漂移)
        z_hat = shape_loop.mask_encoder(m)  # [B,z_dim]
        l_cycle = (z - z_hat.detach()).pow(2).mean()

        # 3) prior (防跑出“超声形状域”)
        l_prior = torch.tensor(0.0, device=device)
        if (prior is not None) and (extract_z_shape_fn is not None):
            zsh = extract_z_shape_fn(z, shape_loop.renderer)  # [B,Dsh]
            nll_raw = prior.nll(zsh, organ_id=None, reduction="none")  # [B]
            nll = nll_raw - nll_raw.detach().amin()  # batch shift

            wpl = float(st.get("wpl", torch.tensor(0.0)).item())
            lam = cfg.lambda0 * (1.0 - wpl)  # 越不自信越依赖 prior
            l_prior = (lam * nll).mean() / cfg.nll_scale

        # 4) 弱图像驱动项：anchor 到 z0（防越训越保守/越圆）
        wpl = float(st.get("wpl", torch.tensor(0.0)).item())
        gamma = cfg.gamma0 * (1.0 - wpl)
        l_anchor = (z - z0.detach()).pow(2).mean()

        # 5) 防“大病灶收缩”：面积保持（很轻的权重）
        area = m.mean(dim=(1, 2, 3))
        l_area = (area - area0).abs().mean()

        J = cfg.w_pl * lpl + l_prior + cfg.beta_cycle * l_cycle + gamma * l_anchor + cfg.w_area * l_area
        J.backward()
        opt.step()

        with torch.no_grad():
            _clamp_z_raw_(z, cfg)

        Jv = float(J.detach().item())
        if Jv < best["J"]:
            best["J"] = Jv
            best["m"] = m.detach()
            best["step"] = k

        logs = {
            "J": Jv,
            "thr": float(thr),
            "pl": float(lpl.detach().item()) if torch.is_tensor(lpl) else float(lpl),
            "cycle": float(l_cycle.detach().item()),
            "prior": float(l_prior.detach().item()),
            "anchor": float(l_anchor.detach().item()),
            "area": float(l_area.detach().item()),
            "pl_ratio": float(st.get("pl_ratio", torch.tensor(0.0)).item()),
            "wpl": float(st.get("wpl", torch.tensor(0.0)).item()),
            "best_step": float(best["step"]),
        }

        if logs["pl_ratio"] < cfg.min_selected_ratio:
            break

    return best["m"], logs
