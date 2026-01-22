import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from dataset.mutlidomain_baseloader import baseloader

from model.Model import ASF

from shape_loop import ShapeClosedLoop

from utils.gmm_prior import GMMPrior, extract_z_shape_from_zraw

# -----------------------------
# small utils
# -----------------------------
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm(x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W] normalized -> [0,1]
    return torch.clamp(x * STD.to(x.device) + MEAN.to(x.device), 0.0, 1.0)


@torch.no_grad()
def dice_binary(pred01: torch.Tensor, gt01: torch.Tensor, eps=1e-6) -> float:
    # pred01, gt01: [1,H,W] or [H,W]
    if pred01.dim() == 3:
        pred01 = pred01.squeeze(0)
    if gt01.dim() == 3:
        gt01 = gt01.squeeze(0)
    pred01 = pred01.float()
    gt01 = gt01.float()
    inter = (pred01 * gt01).sum()
    den = pred01.sum() + gt01.sum() + eps
    return float((2.0 * inter / den).item())


def bce_dice_loss(p: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    p: [B,1,H,W] prob in (0,1)
    y: [B,1,H,W] {0,1}
    w: [B,1,H,W] {0,1} weight mask (confidence region)
    """
    # BCE (masked)
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    bce = (bce * w).sum() / (w.sum() + eps)

    # Dice (masked)
    num = 2.0 * (p * y * w).sum()
    den = (p * w).sum() + (y * w).sum() + eps
    dice = 1.0 - num / den

    return bce + dice


def _topk_bool(score: torch.Tensor, max_ratio: float) -> torch.Tensor:
    """Keep top max_ratio pixels per-sample. score>=0, shape [B,1,H,W]."""
    if max_ratio >= 1.0:
        return torch.ones_like(score, dtype=torch.bool)
    B = score.shape[0]
    flat = score.view(B, -1)
    N = flat.shape[1]
    k = max(1, int(round(max_ratio * N)))
    if k >= N:
        return torch.ones_like(score, dtype=torch.bool)
    topk_vals, _ = torch.topk(flat, k, dim=1, largest=True, sorted=False)
    thr = topk_vals.min(dim=1, keepdim=True)[0]  # [B,1]
    keep = (flat >= thr)
    return keep.view_as(score)

@torch.no_grad()
def build_pl_targets_and_weights(teacher_p, delta_fg, delta_bg, max_ratio_fg=0.03, max_ratio_bg=0.03):
    """
    和 tta_z_only 里的调用保持一致：
    return: y_hat, w_mask(float 0/1), sel_ratio(float), mean_conf(float)
    """
    assert teacher_p.shape[0] == 1, "batch_size=1 only for stable TTA"

    y_hat = (teacher_p > 0.5).float()                    # [1,1,H,W]
    conf  = torch.maximum(teacher_p, 1 - teacher_p)      # [1,1,H,W]

    # FG / BG 分离阈值：BG 更严格
    fg_cand = (y_hat > 0.5) & (conf >= float(delta_fg))
    bg_cand = (y_hat <= 0.5) & (conf >= float(delta_bg))

    H, W = teacher_p.shape[-2], teacher_p.shape[-1]
    total = H * W

    conf_f = conf[0, 0].reshape(-1)
    fg_idx = torch.nonzero(fg_cand[0, 0].reshape(-1), as_tuple=False).squeeze(1)
    bg_idx = torch.nonzero(bg_cand[0, 0].reshape(-1), as_tuple=False).squeeze(1)

    k_fg = min(int(max_ratio_fg * total), int(fg_idx.numel()))
    k_bg = min(int(max_ratio_bg * total), int(bg_idx.numel()))

    keep = []
    if k_fg > 0:
        top = torch.topk(conf_f[fg_idx], k_fg, largest=True).indices
        keep.append(fg_idx[top])
    if k_bg > 0:
        top = torch.topk(conf_f[bg_idx], k_bg, largest=True).indices
        keep.append(bg_idx[top])

    Omega = torch.zeros((1, 1, H, W), dtype=torch.bool, device=teacher_p.device)
    if len(keep) > 0:
        keep = torch.cat(keep, dim=0)
        Omega.view(-1)[keep] = True

    sel_ratio = float(Omega.float().mean().item())
    mean_conf = float(conf[Omega].mean().item()) if Omega.any() else 0.0

    w_mask = Omega.float()  # bce_dice_loss 里当作权重掩码
    return y_hat, w_mask, sel_ratio, mean_conf


def chan_vese_loss(x01: torch.Tensor, m: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    x01: [B,3,H,W] in [0,1]
    m:   [B,1,H,W] prob
    简单 CV：让 mask 内外颜色统计可分（弱图像驱动项）
    """
    m = torch.clamp(m, 0.0, 1.0)
    inside = (m * x01).sum(dim=(2, 3), keepdim=True) / (m.sum(dim=(2, 3), keepdim=True) + eps)
    outside = ((1 - m) * x01).sum(dim=(2, 3), keepdim=True) / ((1 - m).sum(dim=(2, 3), keepdim=True) + eps)

    e_in = ((x01 - inside) ** 2) * m
    e_out = ((x01 - outside) ** 2) * (1 - m)
    return (e_in.mean() + e_out.mean())


def save_white_mask(mask01: torch.Tensor, out_path: str):
    """
    mask01: [H,W] or [1,H,W] or [1,1,H,W], values {0,1} or bool/int
    保存为：黑底 + 白色前景（单通道PNG）
    """
    if mask01.dim() == 4:
        mask01 = mask01[0, 0]
    elif mask01.dim() == 3:
        mask01 = mask01[0]

    m = (mask01 > 0).detach().cpu().numpy().astype(np.uint8) * 255  # 0/255
    Image.fromarray(m, mode="L").save(out_path)





def save_triplet(img01: torch.Tensor, gt01: torch.Tensor, pr01: torch.Tensor, out_path: str):
    """
    img01: [3,H,W] float [0,1]
    gt01/pr01: [H,W] {0,1}
    """
    img = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt = gt01.cpu().numpy().astype(np.uint8)
    pr = pr01.cpu().numpy().astype(np.uint8)

    def overlay(base, mask, color=(255, 0, 0), alpha=0.45):
        base = base.copy()
        m = mask.astype(bool)
        base[m] = (1 - alpha) * base[m] + alpha * np.array(color, dtype=np.float32)
        return base.astype(np.uint8)

    gt_img = overlay(img, gt, (0, 255, 0), 0.45)  # GT green
    pr_img = overlay(img, pr, (255, 0, 0), 0.45)  # Pred red

    gap = 8
    H, W = img.shape[:2]
    canvas = np.zeros((H, W * 3 + gap * 2, 3), dtype=np.uint8)
    canvas[:, :W] = img
    canvas[:, W + gap:W * 2 + gap] = gt_img
    canvas[:, W * 2 + gap * 2:W * 3 + gap * 2] = pr_img
    Image.fromarray(canvas).save(out_path)


# -----------------------------
# TTA core: z-only optimization
# -----------------------------
def tta_z_only(
    model: ASF,
    shape_loop:ShapeClosedLoop,
    prior: GMMPrior,
    x: torch.Tensor,
    steps: int = 5,
    lr_z: float = 5e-2,
    delta_start: float = 0.95,
    delta_end: float = 0.90,
    w_cv: float = 0.15,
    w_cycle: float = 0.20,
    lambda0: float = 0.01,
    lambda_min: float = 0.005,
    w_tr: float = 0.05,
    w_area: float = 0.30,
    pl_max_ratio_fg: float = 0.10,
    pl_max_ratio_bg: float = 0.01,
    bg_thr_bonus: float = 0.05,
    teacher_ema: float = 0.7,

    min_conf_ratio: float = 0.02,
    w_pl: float = 0.15, nll_scale: float = 10.0
):

    device = x.device
    model.eval()

    prior.eval()

    # 1) features + init z0
    with torch.no_grad():
        feats = model.forward_features(x)
        z0, m0, _, _ = shape_loop(feats)
        z0 = z0.detach()
        m0 = m0.detach()
        teacher = m0.clone()
        area0 = (m0 > 0.5).float().mean().detach()
        z_shape0 = extract_z_shape_from_zraw(z0, shape_loop.renderer)
        nll0 = prior.nll(z_shape0, reduction="mean").detach()


    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr_z)

    best = {
        "loss": float("inf"),
        "mask": m0,
        "step": 0,
    }
    logs = []

    WPL_MIN = 0.35

    x01 = denorm(x).detach()  # for CV term

    # ===== prior baseline at z0 (important for batch=1) =====
    with torch.no_grad():
        z_shape0 = extract_z_shape_from_zraw(z0, shape_loop.renderer)
        nll0 = prior.nll(z_shape0, reduction="mean").detach()


    for k in range(steps):
        # delta schedule
        if steps <= 1:
            delta = delta_end
        else:
            t = k / (steps - 1)
            delta = delta_start * (1 - t) + delta_end * t

        opt.zero_grad(set_to_none=True)

        # current mask prob
        m = shape_loop.renderer(z)  # [1,1,H,W], already sigmoid inside

        # --- (A) PL on teacher high-confidence pixels (FG/BG separated) ---
        delta_fg = float(delta)
        delta_bg = float(min(0.99, delta + bg_thr_bonus))

        with torch.no_grad():
            y_hat, w_pl_mask, sel_ratio, mean_conf = build_pl_targets_and_weights(
                teacher_p=teacher,
                delta_fg=delta_fg,
                delta_bg=delta_bg,
                max_ratio_fg=pl_max_ratio_fg,
                max_ratio_bg=pl_max_ratio_bg,
            )

        max_total = max(pl_max_ratio_fg + pl_max_ratio_bg, 1e-6)
        sel_norm = min(sel_ratio / max_total, 1.0)  # 0~1：选满上限 => 1
        wpl = float(mean_conf * sel_norm)  # 0~1：真正的“teacher 可信度”
        reliability = wpl

        # ===== 新增：不可靠就退出，避免“没监督还继续被其它loss带跑” =====
        if (sel_ratio < min_conf_ratio) or (wpl < WPL_MIN):
            logs.append({
                "k": k,
                "skip": True,
                "conf_ratio": float(sel_ratio),
                "wpl": float(wpl),
                "delta": float(delta),
            })
            break

        # 只有可靠时才算PL
        l_pl = bce_dice_loss(m, y_hat, w_pl_mask)


        # m: 当前预测 mask 概率 (0~1), shape [B,1,H,W]
        z_hat = shape_loop.mask_encoder(m)  # soft mask
        l_cycle = F.mse_loss(z, z_hat.detach())

        # gate: confident -> rely more on PL, uncertain -> rely more on prior
        lam = max(lambda_min, lambda0 * (1.0 - reliability))

        # --- (C) prior: penalize getting worse than z0 baseline ---
        z_shape = extract_z_shape_from_zraw(z, shape_loop.renderer)
        nll_now = prior.nll(z_shape, reduction="mean")
        l_prior = F.relu(nll_now - nll0) / nll_scale

        # --- (D) weak image-driven term: Chan-Vese ---
        l_cv = chan_vese_loss(x01, m)

        # --- (E) trust region: keep z near z0 ---
        l_tr = F.mse_loss(z, z0)

        # --- (F) anti-shrink: only penalize becoming smaller than baseline area ---
        area_now = (m > 0.5).float().mean()
        l_area = F.relu(area0 - area_now)

        # gate: confident -> rely more on PL, uncertain -> rely more on prior

        loss = (w_pl * wpl * l_pl) + (lam * l_prior) + (w_cycle * l_cycle) + (w_cv * l_cv) + (w_tr * l_tr) + (w_area * l_area)

        loss.backward()
        opt.step()
        with torch.no_grad():
            teacher.mul_(teacher_ema).add_((1.0 - teacher_ema) * m.detach().clamp(0, 1))

        # safety clamp (raw z)
        with torch.no_grad():
            # z layout: [p, cx, cy, r0, a..., b...]
            z[:, 0].clamp_(-6.0, 6.0)  # p
            z[:, 1:3].clamp_(-2.0, 2.0)  # cx, cy  ★重点：收紧它
            z[:, 3].clamp_(-6.0, 6.0)  # r0
            z[:, 4:].clamp_(-4.0, 4.0)  # a,b

        with torch.no_grad():
            z[:, 1:3].copy_(z0[:, 1:3])  # 彻底禁止平移漂移（cx,cy）

        loss_val = float(loss.item())
        ctr_shift = (z[:, 1:3] - z0[:, 1:3]).norm().item()

        logs.append({
            "k": k,
            "loss": loss_val,
            "delta": float(delta),
            "conf_ratio": sel_ratio,
            "wpl": wpl,
            "l_pl": float(l_pl.item()) if torch.is_tensor(l_pl) else 0.0,
            "l_prior": float(l_prior.item()),
            "l_cycle": float(l_cycle.item()),
            "l_cv": float(l_cv.item()),
            "l_tr": float(l_tr.item()),
            "lam": float(lam),
            "l_area": float(l_area.item()),
            "area_now": float(area_now.item()),
            "ctr_shift": float(ctr_shift),
        })

        if (sel_ratio >= min_conf_ratio) and (wpl >= WPL_MIN) and (loss_val < best["loss"]):
            best["loss"] = loss_val
            best["mask"] = m.detach()
            best["step"] = k + 1

    return best["mask"], logs


def main():
    parser = argparse.ArgumentParser()

    # ckpt / prior
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prior_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="output/tta_eval")

    # data
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--derm_id", type=int, default=3, )

    # TTA hyper
    parser.add_argument("--tta_steps", type=int, default=5)
    parser.add_argument("--lr_z", type=float, default=0.01)
    parser.add_argument("--delta_start", type=float, default=0.97)
    parser.add_argument("--delta_end", type=float, default=0.93)
    parser.add_argument("--w_cv", type=float, default=0.0)
    parser.add_argument("--w_cycle", type=float, default=0.10)
    parser.add_argument("--lambda0", type=float, default=0.02)
    parser.add_argument("--w_tr", type=float, default=0.05)
    parser.add_argument("--z_clip", type=float, default=8.0)
    parser.add_argument("--min_conf_ratio", type=float, default=0.002)

    parser.add_argument("--nll_scale", type=float, default=10.0)
    # baseline switch
    parser.add_argument("--no_tta", action="store_true")

    # data (补齐 baseloader 需要的参数)
    parser.add_argument("--data_configuration",default=r"..\dataset_config.yaml", type=str)

    parser.add_argument("--data_path",default=r"..\my_Dataset",type=str)

    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--w_pl", type=float, default=0.15)
    parser.add_argument("--lambda_min", type=float, default=0.005)
    parser.add_argument("--w_area", type=float, default=0.30)
    parser.add_argument("--pl_max_ratio_fg", type=float, default=0.03)
    parser.add_argument("--pl_max_ratio_bg", type=float, default=0.005)
    parser.add_argument("--bg_thr_bonus", type=float, default=0.05)
    parser.add_argument("--teacher_ema", type=float, default=0.7)
    parser.add_argument("--sharpness", type=float, default=40)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)


    train_loader, val_loader, test_loader = baseloader(args)
    loader = test_loader if test_loader is not None and len(test_loader) > 0 else val_loader

    model = ASF(class_num=2, task_prompt=None, prompt_generator=None, use_anomaly_detection=False).to(device)
    shape_loop = ShapeClosedLoop(out_hw=args.input_size, K=8, sharpness=args.sharpness).to(device)


    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    if isinstance(ckpt, dict) and "shape_loop" in ckpt:
        shape_loop.load_state_dict(ckpt["shape_loop"], strict=True)


    model.eval()
    shape_loop.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in shape_loop.parameters():
        p.requires_grad_(False)


    prior = GMMPrior.load(args.prior_path, device=str(device)).to(device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)


    dices = []
    idx = 0
    for batch in loader:
        img, msk, mk2, setseq, setnam = batch
        img = img.to(device)
        msk = msk.to(device)

        if setseq is not None:
            setseq_t = setseq.to(device)
            keep = (setseq_t == args.derm_id)
            if keep.sum() == 0:
                continue
            img = img[keep]
            msk = msk[keep]


        img = img[:1]
        msk = msk[:1]

        with torch.no_grad():
            feats = model.forward_features(img)
            z0, m0, _, _ = shape_loop(feats)

        if args.no_tta:
            m_best = m0.detach()
            logs = []
        else:
            m_best, logs = tta_z_only(
                model=model,
                shape_loop=shape_loop,
                prior=prior,
                x=img,
                steps=args.tta_steps,
                lr_z=args.lr_z,
                delta_start=args.delta_start,
                delta_end=args.delta_end,
                w_cv=args.w_cv,
                w_cycle=args.w_cycle,
                lambda0=args.lambda0,
                lambda_min=args.lambda_min,
                w_tr=args.w_tr,
                w_area=args.w_area,
                pl_max_ratio_fg=args.pl_max_ratio_fg,
                pl_max_ratio_bg=args.pl_max_ratio_bg,
                bg_thr_bonus=args.bg_thr_bonus,
                teacher_ema=args.teacher_ema,
                z_clip=args.z_clip,
                min_conf_ratio=args.min_conf_ratio,
                w_pl=args.w_pl,
                nll_scale=args.nll_scale,
            )

        pr01 = (m_best[0, 0] > 0.5).long()
        gt01 = (msk[0, 0] > 0.5).long()
        d = dice_binary(pr01, gt01)
        dices.append(d)

        img01 = denorm(img)[0]

        img_np = (img01.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(vis_dir, f"{idx:04d}_input.png"))

        out_path = os.path.join(vis_dir, f"{idx:04d}_dice{d:.3f}_pred.png")
        save_white_mask(pr01, out_path)

        if len(logs) > 0:
            log_path = os.path.join(args.save_dir, "logs.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{idx:04d}] dice={d:.4f} steps={args.tta_steps}\n")
                for row in logs:
                    f.write(str(row) + "\n")
                f.write("\n")

        idx += 1

    mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
    print(f"[DONE] derm samples={len(dices)} mean_dice={mean_dice:.4f}")
    print(f"vis saved to: {vis_dir}")


if __name__ == "__main__":
    main()
