import numpy as np
from PIL import Image

import os
import math
import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset.mutlidomain_baseloader import baseloader
from model.Model import ASF
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR


from shape_loop import ShapeClosedLoop, soft_dice_loss

from utils.gmm_prior import GMMPrior, extract_z_shape_from_zraw

def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float):

    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    for k, tv in t_sd.items():
        sv = s_sd[k]
        if torch.is_floating_point(tv):
            tv.mul_(decay).add_(sv, alpha=1.0 - decay)
        else:
            tv.copy_(sv)

def dice_soft_per(a, b, eps=1e-6):
    # a,b: [B,1,H,W] in [0,1]
    inter = (a * b).sum((1,2,3)) * 2.0
    denom = a.sum((1,2,3)) + b.sum((1,2,3)) + eps
    return inter / denom              # [B]

def dice_soft(a, b, eps=1e-6):
    return dice_soft_per(a, b, eps=eps).mean()

def logit_safe(p, eps=1e-4):
    # p: [0,1] -> logit
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def rampup_sigmoid(x: float):
    # x in [0,1]
    x = float(max(0.0, min(1.0, x)))
    return math.exp(-5.0 * (1.0 - x) * (1.0 - x))

def appearance_aug(x, strength=1.0):
    """
    只做外观扰动：亮度/对比度/噪声/轻微模糊（不做旋转裁剪等几何变换）
    x: [B,C,H,W]
    """
    if strength <= 0:
        return x

    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    # 亮度 & 对比度（逐样本）
    br = (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 0.2 - 0.1) * strength
    ct = (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 0.3 + 0.85) ** strength

    x2 = x * ct + br

    # 高斯噪声
    prob_noise = 0.7 * float(min(1.0, strength))
    if torch.rand(1).item() < prob_noise:
        sigma = (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 0.03) * strength
        x2 = x2 + torch.randn_like(x2) * sigma

    # 轻微模糊（depthwise conv）
    if torch.rand(1).item() < 0.25 * float(min(1.0, strength)):
        k = 3
        # 简单均值核（足够“外观扰动”，不引入几何变化）
        kernel = torch.ones(1, 1, k, k, device=device, dtype=dtype) / (k * k)
        # depthwise: 对每个 channel 独立
        C = x2.shape[1]
        kernel = kernel.repeat(C, 1, 1, 1)
        x2 = F.conv2d(x2, kernel, padding=k//2, groups=C)

    return x2

def random_translate_theta(B, device, max_px_frac=0.08):
    """
    max_px_frac: 相对图像尺寸的最大平移比例(像素域)
    返回 theta, theta_inv: [B,2,3]
    """
    # affine_grid 的平移是在 [-1,1] 归一化坐标系里
    # 像素平移比例 f -> 归一化平移约等于 2f
    t = 2.0 * max_px_frac
    tx = (torch.rand(B, device=device) * 2 - 1) * t
    ty = (torch.rand(B, device=device) * 2 - 1) * t

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    theta_inv = theta.clone()
    theta_inv[:, 0, 2] = -tx
    theta_inv[:, 1, 2] = -ty
    return theta, theta_inv


def random_scale_translate_theta(B, device, max_px_frac=0.08, scale_range=(0.8, 1.2)):
    t = 2.0 * max_px_frac
    tx = (torch.rand(B, device=device) * 2 - 1) * t
    ty = (torch.rand(B, device=device) * 2 - 1) * t
    s  = torch.rand(B, device=device) * (scale_range[1]-scale_range[0]) + scale_range[0]

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    A = torch.zeros(B, 3, 3, device=device)
    A[:, 2, 2] = 1.0
    A[:, :2, :] = theta
    Ainv = torch.inverse(A)
    theta_inv = Ainv[:, :2, :]
    return theta, theta_inv


def warp_with_theta(x, theta, out_hw=None, mode="bilinear", padding_mode="zeros"):
    B, C, H, W = x.shape
    if out_hw is None:
        H2, W2 = H, W
    else:
        H2, W2 = out_hw
    grid = F.affine_grid(theta, size=(B, C, H2, W2), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=False)


# -----------------------
# batch buffer to enforce 1:1
# -----------------------
class MixBuffer:
    def __init__(self):
        self.us = None   # tuple of tensors
        self.derm = None

    def _cat(self, old, new):
        if old is None:
            return new
        return tuple(torch.cat([o, n], dim=0) for o, n in zip(old, new))

    def push(self, IMG, MSK1ch, setseq, derm_id=3):
        # IMG: [B,C,H,W], MSK1ch: [B,1,H,W] or [B,H,W], setseq: [B]
        mask_derm = (setseq == derm_id)
        mask_us = ~mask_derm

        if mask_us.any():
            us_pack = (IMG[mask_us], MSK1ch[mask_us], setseq[mask_us])
            self.us = self._cat(self.us, us_pack)
        if mask_derm.any():
            derm_pack = (IMG[mask_derm], MSK1ch[mask_derm], setseq[mask_derm])
            self.derm = self._cat(self.derm, derm_pack)

    def can_pop(self, us_bs, derm_bs):
        if self.us is None or self.derm is None:
            return False
        return (self.us[0].shape[0] >= us_bs) and (self.derm[0].shape[0] >= derm_bs)

    def pop(self, us_bs, derm_bs):
        def _split(pack, n):
            a = tuple(p[:n] for p in pack)
            b = tuple(p[n:] for p in pack)
            return a, (b if b[0].shape[0] > 0 else None)

        us_take, self.us = _split(self.us, us_bs)
        derm_take, self.derm = _split(self.derm, derm_bs)
        return us_take, derm_take


def _bn_adapt_freeze_affine(m: nn.Module):
    """BN in train mode (update running stats), but freeze affine params.

    UDA / domain shift 下，完全 eval() 的 BN 往往会锁死在 source 统计，DERM 会吃亏；
    这里让 running_mean/var 跟着混合 batch 慢慢适配，同时避免 weight/bias 被训练带偏。
    """
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        m.train()
        # 更小的 momentum 更稳（可选）
        try:
            m.momentum = 0.01
        except Exception:
            pass
        for p in m.parameters():
            p.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=r"..\my_Dataset")
    parser.add_argument("--data_configuration", default=r"..\dataset_config.yaml")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--input_size", default=224, type=int)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--derm_id", default=3, type=int)

    parser.add_argument("--max_epoch", default=201, type=int)
    parser.add_argument("--steps_per_epoch", default=216, type=int)
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--derm_burnin_epochs", default=5, type=int, )


    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)

    parser.add_argument("--stage1_ckpt", default=r"...\best_model.pth", type=str)
    parser.add_argument("--prior_path", type=str)

    # unsup weights
    parser.add_argument("--w_inv", default=0.02, type=float)
    parser.add_argument("--w_cons", default=0.005, type=float)
    parser.add_argument("--w_prior0", default=0.01, type=float)

    parser.add_argument("--w_area", type=float, default=0.25)
    parser.add_argument("--w_cons_bad", type=float, default=0)
    parser.add_argument("--w_prior_bad", type=float, default=0.15)

    # pseudo label
    parser.add_argument("--use_pl", action="store_true", default=False)
    parser.add_argument("--pl_start_epoch", default=10, type=int)
    parser.add_argument("--w_pl", default=0.3, type=float)
    parser.add_argument("--pl_thr", default=0.95, type=float)
    parser.add_argument("--pl_max_ratio", default=0.12, type=float)
    parser.add_argument("--ema_decay", default=0.999, type=float)

    parser.add_argument("--save_dir", default="output/stage2_models")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--fg_lo", type=float, default=0.005,
                        help="foreground ratio lower bound (bin mask) for prior gating")
    parser.add_argument("--fg_hi", type=float, default=0.8,
                        help="foreground ratio upper bound (bin mask) for prior gating")
    parser.add_argument("--fg_hi_area", type=float, default=0.92)
    parser.add_argument("--fg_hi_bad", type=float, default=0.98)

    parser.add_argument("--lam_bad_small", type=float, default=0.05,
                        help="lambda used when fg is too small (all background collapse)")
    parser.add_argument("--lam_bad_big", type=float, default=0.15,
                        help="lambda used when fg is too big (all foreground collapse)")

    parser.add_argument("--lam_max", type=float, default=0.15,
                        help="hard cap for lambda to prevent prior dominating other losses")

    parser.add_argument("--pl_ramp_epochs", default=20, type=int)

    parser.add_argument("--nll_scale", type=float, default=20.0,
                        help="scale down prior nll magnitude (lam*nll)/nll_scale")
    parser.add_argument("--w_unsup", type=float, default=0.8,
                        help="global weight for unsupervised derm loss (multiplies ru)")
    parser.add_argument("--ru_ramp_epochs", type=int, default=30,
                        help="epochs to ramp ru from 0 to 1 after warmup")
    parser.add_argument("--ru_min", type=float, default=0.3,
                        help="initial ru at epoch0 (linearly ramps to 1.0 over ru_ramp_epochs)")
    parser.add_argument("--fg_sharp", type=float, default=40.0)

    parser.add_argument("--sharp_start", type=float, default=40.0, help="renderer sharpness at start")
    parser.add_argument("--sharp_end", type=float, default=120.0, help="renderer sharpness at end")
    parser.add_argument("--sharp_ramp_epochs", type=int, default=30, help="epochs to ramp sharpness")


    parser.add_argument("--lr_backbone", default=2e-5, type=float)

    parser.add_argument("--w_geo", default=0.01, type=float)
    parser.add_argument("--geo_trans", default=0.03, type=float)

    parser.add_argument("--w_ent", default=0.002, type=float)

    parser.add_argument("--ucoef_burnin", type=float, default=0.50)

    parser.add_argument("--w_cv", default=0.15, type=float)

    parser.add_argument("--fg_lo_area", type=float, default=0.02)

    parser.add_argument("--area_scale", type=float, default=120.0)

    parser.add_argument("--uda_ramp_epochs", type=int, default=30)

    parser.add_argument("--cv_big_k", type=float, default=3.0)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--worst_k", type=int, default=8)
    parser.add_argument("--eval_thrs", type=str, default="0.3,0.5,0.7")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = baseloader(args)

    # ---- build student ----
    model = ASF(class_num=2, task_prompt=None, prompt_generator=None, use_anomaly_detection=False).to(device)

    model.disable_decoder_dropout = False

    shape_loop = ShapeClosedLoop(out_hw=args.input_size, K=8, sharpness=args.sharp_start).to(device)


    ckpt = torch.load(args.stage1_ckpt, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        ret = model.load_state_dict(ckpt["model"], strict=False)
    else:
        ret = model.load_state_dict(ckpt, strict=False)


    if isinstance(ckpt, dict) and "shape_loop" in ckpt:
        shape_loop.load_state_dict(ckpt["shape_loop"], strict=True)


    if hasattr(shape_loop, "mask_encoder"):
        set_requires_grad(shape_loop.mask_encoder, False)




    prior = GMMPrior.load(args.prior_path, device=str(device)).to(device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)


    teacher_model = None
    teacher_shape = None
    if args.use_pl:
        teacher_model = copy.deepcopy(model).to(device)
        teacher_shape = copy.deepcopy(shape_loop).to(device)
        teacher_model.eval()

        teacher_shape.eval()
        set_requires_grad(teacher_model, False)
        set_requires_grad(teacher_shape, False)

    # ---- optimizer ----

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_backbone},
            {"params": [p for p in shape_loop.parameters() if p.requires_grad], "lr": args.lr},
        ],
        weight_decay=args.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)



    def _to_rgb_uint8(x_chw: torch.Tensor):
        x = x_chw.detach().float().cpu()
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        mn, mx = x.min(), x.max()
        if (mx - mn) < 1e-6:
            x = torch.zeros_like(x)
        else:
            x = (x - mn) / (mx - mn)
        x = (x * 255.0).clamp(0, 255).byte()
        return x.permute(1, 2, 0).numpy()  # HWC uint8

    def _overlay(rgb_uint8: np.ndarray, mask01_hw: np.ndarray, alpha=0.45):

        out = rgb_uint8.copy()
        m = mask01_hw.astype(bool)
        out[m, 0] = (out[m, 0] * (1 - alpha) + 255 * alpha).astype(np.uint8)
        out[m, 1] = (out[m, 1] * (1 - alpha)).astype(np.uint8)
        out[m, 2] = (out[m, 2] * (1 - alpha)).astype(np.uint8)
        return out

    @torch.no_grad()
    def visualize_us_and_derm(epoch, img_us, msk_us, img_d, msk_d, save_root):
        model.eval()
        shape_loop.eval()

        os.makedirs(save_root, exist_ok=True)


        feats_us = model.forward_features(img_us.to(device))
        _, m_us, _, _ = shape_loop(feats_us)
        msk_us = msk_us.to(device).long()
        if msk_us.ndim == 4:
            msk_us = msk_us.squeeze(1)
        gt_us = (msk_us > 0).float().unsqueeze(1)
        m_us_up = m_us
        if m_us_up.shape[-2:] != gt_us.shape[-2:]:
            m_us_up = F.interpolate(m_us_up, size=gt_us.shape[-2:], mode="bilinear", align_corners=False)

        dice_us = dice_soft_per((m_us_up > 0.5).float(), gt_us).detach().cpu().numpy()


        img_d = img_d.to(device)
        feats_d = model.forward_features(img_d)
        _, m_d, _, _ = shape_loop(feats_d)

        x_a = appearance_aug(img_d, strength=0.6)
        x_b = appearance_aug(img_d, strength=0.3)
        _, m_a, _, _ = shape_loop(model.forward_features(x_a))
        _, m_b, _, _ = shape_loop(model.forward_features(x_b))

        msk_d = msk_d.to(device).long()

        if msk_d.ndim == 4:
            msk_d = msk_d.squeeze(1)


        gt_d  = (msk_d  > 0).float().unsqueeze(1)



        m_d_up = m_d
        if m_d_up.shape[-2:] != gt_d.shape[-2:]:
            m_d_up = F.interpolate(m_d_up, size=gt_d.shape[-2:], mode="bilinear", align_corners=False)

        dice_d = dice_soft_per((m_d_up > 0.5).float(), gt_d).detach().cpu().numpy()


        m_a_up, m_b_up = m_a, m_b
        if m_a_up.shape[-2:] != gt_d.shape[-2:]:
            m_a_up = F.interpolate(m_a_up, size=gt_d.shape[-2:], mode="bilinear", align_corners=False)
        if m_b_up.shape[-2:] != gt_d.shape[-2:]:
            m_b_up = F.interpolate(m_b_up, size=gt_d.shape[-2:], mode="bilinear", align_corners=False)


        dice_a = dice_soft_per((m_a_up > 0.5).float(), gt_d).detach().cpu().numpy()
        dice_b = dice_soft_per((m_b_up > 0.5).float(), gt_d).detach().cpu().numpy()


        keep_thr = 0.80
        max_save = 4

        robust = (dice_d >= keep_thr) & (dice_a >= keep_thr) & (dice_b >= keep_thr)
        robust_idx = np.where(robust)[0].tolist()


        if len(robust_idx) < max_save:
            score = np.minimum(np.minimum(dice_d, dice_a), dice_b)
            order = np.argsort(-score)
            for j in order.tolist():
                if j not in robust_idx:
                    robust_idx.append(j)
                if len(robust_idx) >= max_save:
                    break
        else:
            robust_idx = robust_idx[:max_save]


        from pathlib import Path


        save_root = Path(save_root)
        us_dir = save_root / "US_single"
        derm_dir = save_root / "DERM_single"
        us_dir.mkdir(parents=True, exist_ok=True)
        derm_dir.mkdir(parents=True, exist_ok=True)


        B_us = min(4, img_us.shape[0])
        for i in range(B_us):
            us_img = _to_rgb_uint8(img_us[i])
            us_gt_img = _overlay(us_img, gt_us[i, 0].cpu().numpy() > 0.5)
            us_pd_img = _overlay(us_img, (m_us_up[i, 0].detach().cpu().numpy() > 0.5))

            base = us_dir / f"epoch{epoch:03d}_US_{i:02d}_dice{dice_us[i]:.3f}"
            Image.fromarray(us_img).save(str(base) + "_orig.png")
            Image.fromarray(us_gt_img).save(str(base) + "_GT.png")
            Image.fromarray(us_pd_img).save(str(base) + "_Pred.png")


        if len(robust_idx) == 0:
            score = np.minimum(np.minimum(dice_d, dice_a), dice_b)
            robust_idx = [int(np.argmax(score))]

        for rank, i in enumerate(robust_idx):

            d_img0 = _to_rgb_uint8(img_d[i])
            d_imga = _to_rgb_uint8(x_a[i])
            d_imgb = _to_rgb_uint8(x_b[i])

            gt01 = (gt_d[i, 0].cpu().numpy() > 0.5)


            d_gt0 = _overlay(d_img0, gt01)
            d_pd0 = _overlay(d_img0, (m_d_up[i, 0].detach().cpu().numpy() > 0.5))


            d_gta = _overlay(d_imga, gt01)
            d_pda = _overlay(d_imga, (m_a_up[i, 0].detach().cpu().numpy() > 0.5))


            d_gtb = _overlay(d_imgb, gt01)
            d_pdb = _overlay(d_imgb, (m_b_up[i, 0].detach().cpu().numpy() > 0.5))

            base = derm_dir / (
                f"epoch{epoch:03d}_DERM_r{rank:02d}_idx{i:02d}_"
                f"d{dice_d[i]:.3f}_a{dice_a[i]:.3f}_b{dice_b[i]:.3f}"
            )


            Image.fromarray(d_img0).save(str(base) + "_orig.png")
            Image.fromarray(d_gt0).save(str(base) + "_orig_GT.png")
            Image.fromarray(d_pd0).save(str(base) + "_orig_Pred.png")

            Image.fromarray(d_imga).save(str(base) + "_augA.png")
            Image.fromarray(d_gta).save(str(base) + "_augA_GT.png")
            Image.fromarray(d_pda).save(str(base) + "_augA_Pred.png")

            Image.fromarray(d_imgb).save(str(base) + "_augB.png")
            Image.fromarray(d_gtb).save(str(base) + "_augB_GT.png")
            Image.fromarray(d_pdb).save(str(base) + "_augB_Pred.png")

        us_avg = float(dice_us[:B_us].mean()) if B_us > 0 else float("nan")
        if len(robust_idx) > 0:
            derm_avg = float(np.mean([dice_d[j] for j in robust_idx]))
        else:
            derm_avg = float(dice_d.mean()) if len(dice_d) > 0 else float("nan")



    def _parse_thrs(s: str):
        return [float(x) for x in s.split(",") if len(x.strip()) > 0]

    @torch.no_grad()
    def eval_full_val(epoch: int, save_root: str):

        model.eval()
        shape_loop.eval()

        thrs = _parse_thrs(args.eval_thrs)
        main_thrs = [0.3, 0.5]

        os.makedirs(save_root, exist_ok=True)


        epoch_dir = os.path.join(save_root, f"epoch{epoch:03d}")
        worst_us_dir = os.path.join(epoch_dir, "worst_us")  # 目前未用到，但保留目录
        os.makedirs(worst_us_dir, exist_ok=True)

        worst_derm_dir = {t: os.path.join(epoch_dir, f"worst_derm_thr{t:.2f}") for t in main_thrs}
        for t in main_thrs:
            os.makedirs(worst_derm_dir[t], exist_ok=True)


        us_dices = {t: [] for t in thrs}
        derm_dices = {t: [] for t in thrs}

        derm_bucket = {t: {"small": [], "med": [], "large": []} for t in main_thrs}
        worst_derm = {t: [] for t in main_thrs}

        def _push_worst(buf, dice_val, rgb, gt01, pd01, tag, k):
            buf.append((dice_val, rgb, gt01, pd01, tag))
            buf.sort(key=lambda x: x[0])  # small->large (keep worst)
            if len(buf) > k:
                buf.pop()

        for batch in val_loader:
            IMG, MSK1ch, _, setseq, _ = batch
            IMG = IMG.to(device, non_blocking=True)
            MSK1ch = MSK1ch.to(device, non_blocking=True)
            setseq = setseq.to(device, non_blocking=True).long()


            mask_us = (setseq != args.derm_id)
            if mask_us.any():
                img = IMG[mask_us]
                msk = MSK1ch[mask_us]
                if msk.ndim == 4:
                    msk = msk.squeeze(1)
                gt = (msk > 0).float().unsqueeze(1)  # [B,1,H,W]

                feats = model.forward_features(img)
                _, m, _, _ = shape_loop(feats)  # [B,1,h,w]
                if m.shape[-2:] != gt.shape[-2:]:
                    m = F.interpolate(m, size=gt.shape[-2:], mode="bilinear", align_corners=False)

                for t in thrs:
                    d = dice_soft_per((m > t).float(), gt).detach().cpu().numpy().tolist()
                    us_dices[t].extend(d)

            mask_d = (setseq == args.derm_id)
            if mask_d.any():
                img = IMG[mask_d]
                msk = MSK1ch[mask_d]
                if msk.ndim == 4:
                    msk = msk.squeeze(1)
                gt = (msk > 0).float().unsqueeze(1)  # [B,1,H,W]
                fg_ratio = gt.mean(dim=(1, 2, 3)).detach().cpu().numpy()  # [B]

                feats = model.forward_features(img)
                _, m, _, _ = shape_loop(feats)
                if m.shape[-2:] != gt.shape[-2:]:
                    m = F.interpolate(m, size=gt.shape[-2:], mode="bilinear", align_corners=False)

                # 1) thr 曲线
                for t0 in thrs:
                    d0 = dice_soft_per((m > t0).float(), gt).detach().cpu().numpy().tolist()
                    derm_dices[t0].extend(d0)

                # 2) buckets + worst_k (只对 main_thrs)
                for t1 in main_thrs:
                    pd = (m > t1).float()
                    d1 = dice_soft_per(pd, gt).detach().cpu().numpy()  # [B]
                    for i in range(len(d1)):
                        fr = float(fg_ratio[i])
                        if fr < 0.25:
                            bucket = "small"
                        elif fr < 0.40:
                            bucket = "med"
                        else:
                            bucket = "large"
                        derm_bucket[t1][bucket].append(float(d1[i]))

                        rgb = _to_rgb_uint8(img[i])  # HWC uint8
                        gt01 = (gt[i, 0].detach().cpu().numpy() > 0.5)
                        pd01 = (pd[i, 0].detach().cpu().numpy() > 0.5)
                        tag = f"{bucket}_{i}"
                        _push_worst(worst_derm[t1], float(d1[i]), rgb, gt01, pd01, tag, args.worst_k)

        def _summ(x):
            if len(x) == 0:
                return "NA"
            a = np.array(x, dtype=np.float32)
            return f"mean={a.mean():.3f} med={np.median(a):.3f} p10={np.percentile(a, 10):.3f} p90={np.percentile(a, 90):.3f} (n={len(a)})"

        for t in thrs:
            print(f"[VAL] thr={t:.2f} | US {_summ(us_dices[t])} | DERM {_summ(derm_dices[t])}")

        for t in main_thrs:
            print(
                f"[VAL][DERM buckets @thr={t:.2f}] "
                f"small({_summ(derm_bucket[t]['small'])}) | "
                f"med({_summ(derm_bucket[t]['med'])}) | "
                f"large({_summ(derm_bucket[t]['large'])})"
            )

        for t in main_thrs:
            for rank, (d, rgb, gt01, pd01, tag) in enumerate(worst_derm[t]):
                img0 = rgb
                gt0 = _overlay(img0, gt01)
                pd0 = _overlay(img0, pd01)
                panel = Image.fromarray(np.concatenate([img0, gt0, pd0], axis=1))
                panel.save(os.path.join(worst_derm_dir[t], f"{rank:02d}_dice{d:.3f}_{tag}.png"))

        # ---------- return metrics ----------
        def _mean_or_nan(x):
            return float(np.mean(x)) if len(x) > 0 else float("nan")

        metrics = {}
        for t in thrs:
            metrics[f"us_mean_{t:.2f}"] = _mean_or_nan(us_dices[t])
            metrics[f"derm_mean_{t:.2f}"] = _mean_or_nan(derm_dices[t])
        for t in main_thrs:
            metrics[f"derm_small_{t:.2f}"] = _mean_or_nan(derm_bucket[t]["small"])
            metrics[f"derm_med_{t:.2f}"]   = _mean_or_nan(derm_bucket[t]["med"])
            metrics[f"derm_large_{t:.2f}"] = _mean_or_nan(derm_bucket[t]["large"])

        return metrics

    def supervised_us_losses(img_us, msk1ch_us, freeze_mask_encoder=False, detach_backbone=False):

        img_us = img_us.to(device, non_blocking=True)
        msk1ch_us = msk1ch_us.to(device, non_blocking=True).long()
        if msk1ch_us.ndim == 4:
            msk1ch_us = msk1ch_us.squeeze(1)  # [B,H,W]

        msk_bin = (msk1ch_us > 0).long()

        gt = msk_bin.float().unsqueeze(1)  # [B,1,H,W]

        if detach_backbone:
            with torch.no_grad():
                feats = model.forward_features(img_us)
        else:
            feats = model.forward_features(img_us)

        z1, m1, z2, m2 = shape_loop(feats)


        if m1.shape[-2:] != gt.shape[-2:]:
            m1_up = F.interpolate(m1, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        else:
            m1_up = m1

        term_seg = F.binary_cross_entropy(m1_up.clamp(1e-6, 1-1e-6), gt) + soft_dice_loss(m1_up, gt)


        z_gt = shape_loop.mask_encoder(gt)
        term_cons = F.mse_loss(z1, z_gt.detach())

        if freeze_mask_encoder:

            loss_us = term_seg + 0.1 * term_cons
            stats = {
                "us_seg": term_seg.detach(),
                "us_cons": term_cons.detach(),
                "us_rec": torch.tensor(0.0, device=device),
                "us_gt_rec": torch.tensor(0.0, device=device),
            }
            return loss_us, stats


        m_gt = shape_loop.renderer(z_gt)

        if m_gt.shape[-2:] != gt.shape[-2:]:
            m_gt = F.interpolate(m_gt, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        term_gt_rec = (
                F.binary_cross_entropy(m_gt.clamp(1e-6, 1 - 1e-6), gt) +
                soft_dice_loss(m_gt, gt)
        )

        m1_det = (m1.detach() > 0.5).float()

        if m2.shape[-2:] != m1_det.shape[-2:]:
            m2 = F.interpolate(m2, size=m1_det.shape[-2:], mode="bilinear", align_corners=False)

        term_rec = F.binary_cross_entropy(m2.clamp(1e-6, 1 - 1e-6), m1_det) + soft_dice_loss(m2, m1_det)

        fp = (m1_up * (1 - gt)).mean()
        loss_us = term_seg + 0.1*term_cons + 0.5*term_rec + 1.0*term_gt_rec + 0.2*fp
        stats = {
            "us_seg": term_seg.detach(),
            "us_cons": term_cons.detach(),
            "us_rec": term_rec.detach(),
            "us_gt_rec": term_gt_rec.detach(),
        }
        return loss_us, stats


    def chan_vese_loss(img, mask, eps=1e-6):

        I = img  # (B,3,H,W)  用于 c1/c0、pix_data
        Ig = img.mean(dim=1, keepdim=True)  # (B,1,H,W)  专门给边缘项用

        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

        C = img.shape[1]
        kxC = kx.repeat(C, 1, 1, 1)
        kyC = ky.repeat(C, 1, 1, 1)
        gx = F.conv2d(img, kxC, padding=1, groups=C)  # [B,C,H,W]
        gy = F.conv2d(img, kyC, padding=1, groups=C)
        g = torch.sqrt((gx * gx + gy * gy).sum(dim=1, keepdim=True) + 1e-6)  # [B,1,H,W]
        g = g / (g.mean(dim=(2, 3), keepdim=True) + 1e-6)
        w_edge = g.clamp(0.0, 5.0).detach()  # [B,1,H,W]

        m = mask.clamp(0.0, 1.0)
        m_sum = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        bg_sum = (1.0 - m).sum(dim=(2, 3), keepdim=True).clamp_min(eps)

        c1 = (I * m).sum(dim=(2, 3), keepdim=True) / m_sum
        c0 = (I * (1.0 - m)).sum(dim=(2, 3), keepdim=True) / bg_sum

        loss_in = ((I - c1) ** 2 * m * w_edge).mean(dim=(1, 2, 3))  # [B]
        loss_out = ((I - c0) ** 2 * (1.0 - m) * w_edge).mean(dim=(1, 2, 3))  # [B]
        loss_per = loss_in + loss_out


        g_stop = (1.0 / (1.0 + 10.0 * w_edge)).detach()  # [B,1,H,W]

        m = mask.clamp(0.0, 1.0)
        mx = F.pad(m[:, :, :, 1:] - m[:, :, :, :-1], (0, 1, 0, 0))
        my = F.pad(m[:, :, 1:, :] - m[:, :, :-1, :], (0, 0, 0, 1))
        grad_m = torch.sqrt(mx * mx + my * my + 1e-12)  # [B,1,H,W]

        len_per = (g_stop * grad_m).mean(dim=(1, 2, 3))  # [B]
        loss_per = loss_per + 0.20 * len_per

        return loss_per


    def deep_chan_vese_loss(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

        if feat.shape[-2:] != mask.shape[-2:]:
            m = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        else:
            m = mask

        m = m.clamp(0.0, 1.0)
        inv = 1.0 - m

        m_sum = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        inv_sum = inv.sum(dim=(2, 3), keepdim=True).clamp_min(eps)

        # per-channel region means
        c1 = (feat * m).sum(dim=(2, 3), keepdim=True) / m_sum
        c0 = (feat * inv).sum(dim=(2, 3), keepdim=True) / inv_sum

        # average over (C,H,W) => per-sample
        loss_in = ((feat - c1) ** 2 * m).mean(dim=(1, 2, 3))
        loss_out = ((feat - c0) ** 2 * inv).mean(dim=(1, 2, 3))
        return loss_in + loss_out

    def derm_unsup_losses(img_d, backbone_grad=False, only_data=False, epoch=0):

        img_d = img_d.to(device, non_blocking=True)

        # two appearance views (a stronger, b weaker)
        x_data = img_d             # CV/area/prior 用原图最稳
        x_a = appearance_aug(img_d, strength=0.6)
        x_b = appearance_aug(img_d, strength=0.3)

        # 用 x_data 走一个前向得到 m_data（替代 m1b 做 data term）
        if backbone_grad:
            feats_data = model.forward_features(x_data)
        else:
            with torch.no_grad():
                feats_data = model.forward_features(x_data)
        z1d, m1d, _, _ = shape_loop(feats_data)

        if backbone_grad:
            feats_a = model.forward_features(x_a)
        else:
            with torch.no_grad():
                feats_a = model.forward_features(x_a)
        z1a, m1a, _, _ = shape_loop(feats_a)

        if backbone_grad:
            feats_b = model.forward_features(x_b)
        else:
            with torch.no_grad():
                feats_b = model.forward_features(x_b)
        z1b, m1b, _, _ = shape_loop(feats_b)

        p_det = m1d.detach().clamp(1e-4, 1.0 - 1e-4)
        conf_pix = torch.maximum(p_det, 1.0 - p_det)  # [B,1,h,w]

        band = ((p_det > 0.2) & (p_det < 0.8)).float()
        band_sum = band.sum(dim=(1, 2, 3))  # [B]

        conf_band = (conf_pix * band).sum(dim=(1, 2, 3)) / band_sum.clamp_min(1.0)
        conf_full = conf_pix.mean(dim=(1, 2, 3))
        conf = torch.where(band_sum > 0, conf_band, conf_full)  # [B]

        # agreement-based confidence (downweights inconsistent predictions under aug)
        agree = 1.0 - (m1a.detach() - m1b.detach()).abs().mean(dim=(1, 2, 3))
        conf = (0.5 * conf + 0.5 * agree.clamp(0.0, 1.0)).clamp(0.0, 1.0)

        w_conf = (1.0 - conf).detach()

        fg_softbin = torch.sigmoid((m1d-0.5) * args.fg_sharp).mean(dim=(1, 2, 3))  # [B]  (with grad)
        fg_det = fg_softbin.detach()  # [B]

        bad_small = (fg_det < args.fg_lo)
        bad_big = (fg_det > args.fg_hi_bad)
        bad = bad_small | bad_big


        if args.w_geo > 0:
            B = x_b.shape[0]
            theta, theta_inv = random_translate_theta(B, x_b.device, max_px_frac=args.geo_trans)

            # 图像平移：用 border padding，避免黑边
            x_g = warp_with_theta(x_b, theta, out_hw=x_b.shape[-2:], padding_mode="border")

            if backbone_grad:
                feats_g = model.forward_features(x_g)
            else:
                with torch.no_grad():
                    feats_g = model.forward_features(x_g)
            _, m_g, _, _ = shape_loop(feats_g)

            ones = torch.ones((B, 1, x_b.shape[-2], x_b.shape[-1]), device=x_b.device)
            valid_g = warp_with_theta(ones, theta, out_hw=x_b.shape[-2:], padding_mode="zeros")
            valid_back = warp_with_theta(valid_g, theta_inv, out_hw=m1b.shape[-2:], padding_mode="zeros")
            w = (valid_back > 0.999).float()  # [B,1,h,w]

            m_g_back = warp_with_theta(m_g, theta_inv, out_hw=m1b.shape[-2:], padding_mode="zeros")

            log_b_geo = logit_safe(m1b)
            log_gb = logit_safe(m_g_back)

            w_sum = w.sum((1, 2, 3)).clamp_min(1.0)  # [B]
            l_geo_per = 0.5 * (
                    ((log_b_geo - log_gb.detach()).abs() * w).sum((1, 2, 3)) / w_sum +
                    ((log_gb - log_b_geo.detach()).abs() * w).sum((1, 2, 3)) / w_sum
            )
            l_geo = (w_conf * l_geo_per).mean()
        else:
            l_geo = torch.tensor(0.0, device=device)


        zsh_d = extract_z_shape_from_zraw(z1d, shape_loop.renderer)
        zsh_b = extract_z_shape_from_zraw(z1b, shape_loop.renderer)

        inv_per = 0.5 * (
            (zsh_d - zsh_b.detach()).pow(2).mean(dim=1) +
            (zsh_b - zsh_d.detach()).pow(2).mean(dim=1)
        )  # [B]


        p_d, cx_d, cy_d, _, _, _ = shape_loop.renderer.decode(z1d)
        p_b, cx_b, cy_b, _, _, _ = shape_loop.renderer.decode(z1b)
        pose_d = torch.cat([p_d, cx_d, cy_d], dim=1)   # [B,3]
        pose_b = torch.cat([p_b, cx_b, cy_b], dim=1)
        pose_per = 0.5 * (
            (pose_d - pose_b.detach()).pow(2).mean(dim=1) +
            (pose_b - pose_d.detach()).pow(2).mean(dim=1)
        )

        conf_w = ((conf - 0.5) / 0.5).clamp(0.0, 1.0).detach()
        conf_w = conf_w.clamp(min=0.3)
        w_inv = torch.where(bad, torch.full_like(inv_per, args.w_cons_bad), torch.ones_like(inv_per))

        l_inv = (w_inv * conf_w * (inv_per + 0.1 * pose_per)).mean()


        def _dice_cons_per(a, b, eps=1e-6):
            inter = (a * b).sum(dim=(1, 2, 3)) * 2.0
            denom = a.sum(dim=(1, 2, 3)) + b.sum(dim=(1, 2, 3)) + eps
            return 1.0 - inter / denom

        l_cons_per = 0.5 * (_dice_cons_per(m1a, m1b.detach()) + _dice_cons_per(m1b, m1a.detach()))


        w_cons = torch.where(
            bad,
            torch.full_like(l_cons_per, args.w_cons_bad),
            torch.ones_like(l_cons_per)
        )
        l_cons = (w_cons * conf_w * l_cons_per).mean()


        p = m1b.clamp(1e-6, 1.0 - 1e-6)
        ent_per = (-(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))).mean(dim=(1, 2, 3))  # [B]
        w_ent = torch.where(bad, torch.zeros_like(ent_per), torch.ones_like(ent_per))  # bad样本不加这项，避免把坍塌“硬化”
        l_ent = (w_ent * ent_per).mean()


        apply_lb = (fg_det < 0.003).float()
        lb = apply_lb * args.fg_lo_area
        l_area = (F.relu(lb - fg_softbin) + F.relu(fg_softbin - args.fg_hi_area)).mean() * args.area_scale


        lam = args.w_prior_bad * bad_big.float()


        mid_small = ((fg_det >= args.fg_lo) & (fg_det <= 0.25)).float()
        lam = lam + (args.w_prior0 * (1.0 - conf) * mid_small)


        lam = torch.where(bad_big, torch.maximum(lam, torch.full_like(lam, args.lam_bad_big)), lam)
        lam = lam.clamp(max=args.lam_max)


        nll_raw = prior.nll(zsh_d, organ_id=None, reduction="none")

        nll = nll_raw - nll_raw.detach().amin()
        logp_mean = (-nll_raw).mean().detach()
        nll_mean = nll.mean().detach()
        l_prior = (lam * nll).mean() / args.nll_scale


        cv_pix_per = chan_vese_loss(x_data, m1d)
        cv_feat_per = deep_chan_vese_loss(feats_data[1], m1d)  # f3
        cv_per = 0.8 * cv_pix_per + 0.2 * cv_feat_per


        fg_big_start = 0.35
        w_big = 1.0 + args.cv_big_k * ((fg_det - fg_big_start).clamp(min=0.0) / (1.0 - fg_big_start))
        w_big = w_big.clamp(max=1.0 + args.cv_big_k)  # safety cap
        l_cv = (w_big * cv_per).mean()

        stats = {
            "inv": l_inv.detach(),
            "cons": l_cons.detach(),
            "prior": l_prior.detach(),
            "conf_mean": conf.mean().detach(),
            "lam_mean": lam.mean().detach(),
            "fg_mean": fg_det.mean().detach(),
            "bad_ratio": bad.float().mean().detach(),
            "geo": l_geo.detach(),
            "nll_mean": nll_mean,
            "logp_mean": logp_mean,
            "ent": l_ent.detach(),
            "cv": l_cv.detach(),
            "area": l_area.detach(),
            "bad_small_ratio": bad_small.float().mean().detach(),
            "bad_big_ratio": bad_big.float().mean().detach(),
            "w_cons_mean": w_cons.mean().detach(),
        }


        if only_data:

            loss_unsup = args.w_cv * l_cv + args.w_area * l_area
        else:

            t = (epoch - args.derm_burnin_epochs + 1) / max(1, args.uda_ramp_epochs)
            r = float(max(0.0, min(1.0, t)))

            loss_unsup = (
                    args.w_cv * l_cv
                    + args.w_area * l_area
                    + r * (args.w_inv * l_inv + args.w_cons * l_cons + l_prior + args.w_geo * l_geo + args.w_ent * l_ent)
            )

        return loss_unsup, stats, (x_a, x_b, m1a)

    def pseudo_label_loss(img_strong, img_weak):

        if (teacher_model is None) or (teacher_shape is None):
            return torch.tensor(0.0, device=device), {"pl": torch.tensor(0.0, device=device), "pl_ratio": torch.tensor(0.0, device=device)}

        with torch.no_grad():
            feats_t = teacher_model.forward_features(img_weak)
            _, mT, _, _ = teacher_shape(feats_t)
            prob = mT.clamp(0.0, 1.0)




            fg_ratio = (prob > 0.5).float().mean(dim=(1, 2, 3))  # [B]


            valid = (fg_ratio >= 0.001) & (fg_ratio <= 0.95)


            valid4 = valid.float().view(-1, 1, 1, 1)



            conf_pix = torch.maximum(prob, 1.0 - prob)  # [B,1,H,W]


            w0 = ((conf_pix - args.pl_thr) / (1.0 - args.pl_thr)).clamp(0.0, 1.0)

            if args.pl_max_ratio < 1.0:
                B = w0.shape[0]
                N = w0[0].numel()
                k = int(max(1, min(N, round(args.pl_max_ratio * N))))
                flat = w0.view(B, -1)
                topv, topi = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
                sel = torch.zeros_like(flat)
                sel.scatter_(1, topi, 1.0)
                w0 = (flat * sel).view_as(w0)


            y = (prob > 0.5).float()


            w0 = w0 * valid4
            y  = y  * valid4


            w_pos = w0 * y
            w_neg = w0 * (1.0 - y)

            sum_pos = w_pos.sum((1, 2, 3)).clamp_min(1.0)
            sum_neg = w_neg.sum((1, 2, 3)).clamp_min(1.0)

            scale = (sum_neg / sum_pos).clamp(1.0, 10.0).view(-1, 1, 1, 1)
            w = w_pos * scale + w_neg



            ratio = ((w0 > 0).float().sum((1,2,3)) / float(w0[0].numel())).mean().detach()

        feats_s = model.forward_features(img_strong)
        _, mS, _, _ = shape_loop(feats_s)

        if w.shape[-2:] != mS.shape[-2:]:
            w = F.interpolate(w, size=mS.shape[-2:], mode="nearest")
            y = F.interpolate(y, size=mS.shape[-2:], mode="nearest")
            y = (y > 0.5).float()

        bce_map = F.binary_cross_entropy(mS.clamp(1e-6, 1 - 1e-6), y, reduction="none")

        wsum = w.sum((1, 2, 3))

        has = (wsum > 0).float()  # [B]

        if has.sum() < 0.5:

            z = torch.tensor(0.0, device=device)

            return z, {"pl": z, "pl_ratio": ratio}


        bce = (bce_map * w).sum((1, 2, 3)) / wsum.clamp_min(1.0)

        bce = (bce * has).sum() / has.sum().clamp_min(1.0)




        inter = ((mS * y) * w).sum((1, 2, 3)) * 2.0

        denom2 = ((mS * w).sum((1, 2, 3)) + (y * w).sum((1, 2, 3)) + 1e-6)

        dice_per = inter / denom2

        dice = (dice_per * has).sum() / has.sum().clamp_min(1.0)

        ldice = 1.0 - dice


        lpl = bce + ldice


        return lpl, {"pl": lpl.detach(), "pl_ratio": ratio}

    best_score = -1e9
    best_epoch = -1
    best_path = Path(args.save_dir) / "stage2_best.pth"

    for epoch in range(args.max_epoch):
        model.train()
        shape_loop.train()

        if epoch < args.warmup_epoch:
            set_requires_grad(model, False)
            set_requires_grad(shape_loop, True)
        else:

            set_requires_grad(model, False)

            keys = ["layer4", "layer3", "bicst_34", "bicst_23", "backbone.layer4", "backbone.layer3"]
            for n, p in model.named_parameters():
                if any(k in n for k in keys):
                    p.requires_grad_(True)


            set_requires_grad(shape_loop, True)


        freeze_mask = True
        if hasattr(shape_loop, "mask_encoder"):
            set_requires_grad(shape_loop.mask_encoder, False)


        model.apply(_bn_adapt_freeze_affine)
        shape_loop.apply(_bn_adapt_freeze_affine)

        if args.sharp_ramp_epochs > 0:
            t_sh = min(1.0, epoch / float(args.sharp_ramp_epochs))
            cur_sharp = args.sharp_start + (args.sharp_end - args.sharp_start) * t_sh
        else:
            cur_sharp = args.sharp_end
        try:
            shape_loop.renderer.sharpness = float(cur_sharp)
        except Exception:
            pass
        if args.use_pl and (teacher_shape is not None):
            try:
                teacher_shape.renderer.sharpness = float(cur_sharp)
            except Exception:
                pass


        mixbuf = MixBuffer()
        it = iter(train_loader)

        pbar = tqdm(total=args.steps_per_epoch, desc=f"Stage2 Epoch {epoch}")
        step = 0

        loss_meter = {
            "us": 0.0, "unsup": 0.0, "pl": 0.0,
            "inv": 0.0, "cons": 0.0, "prior": 0.0, "nll_mean": 0.0,
            "logp": 0.0,
            "conf": 0.0, "lam": 0.0, "pl_ratio": 0.0,
            "ru": 0.0,
            "fg": 0.0,
            "bad_ratio": 0.0,
            "rp": 0.0,
            "area": 0.0,

            "cv": 0.0,
            "bad_small": 0.0,
            "bad_big": 0.0,
            "geo": 0.0,
            "w_cons": 0.0
        }

        us_bs = args.batch_size // 2
        derm_bs = args.batch_size - us_bs

        while step < args.steps_per_epoch:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            IMG, MSK1ch, MSKmul, setseq, extra = batch
            IMG = IMG.to(device, non_blocking=True)
            MSK1ch = MSK1ch.to(device, non_blocking=True)
            setseq = setseq.to(device, non_blocking=True).long()

            mixbuf.push(IMG, MSK1ch, setseq, derm_id=args.derm_id)

            while mixbuf.can_pop(us_bs=us_bs, derm_bs=derm_bs) and (step < args.steps_per_epoch):
                (img_us, msk_us, _), (img_d, _msk_d, _sd) = mixbuf.pop(us_bs=us_bs, derm_bs=derm_bs)


                detach_us_backbone = False

                loss_us, st_us = supervised_us_losses(
                    img_us, msk_us,
                    freeze_mask_encoder=freeze_mask,
                    detach_backbone=detach_us_backbone
                )


                if args.ru_ramp_epochs <= 1:
                    ru = 1.0
                else:
                    t_ru = min(1.0, float(epoch + 1) / float(args.ru_ramp_epochs))
                    ru = args.ru_min + (1.0 - args.ru_min) * t_ru


                derm_backbone_grad = (epoch >= max(args.warmup_epoch, args.derm_burnin_epochs))

                loss_unsup, st_u, (x_a, x_b, _m1a) = derm_unsup_losses(img_d, backbone_grad=derm_backbone_grad, only_data=(epoch < args.derm_burnin_epochs), epoch=epoch)

                if epoch < args.derm_burnin_epochs:
                    ucoef = args.ucoef_burnin
                else:
                    t_u = min(1.0, float(epoch - args.derm_burnin_epochs + 1) / float(args.uda_ramp_epochs))
                    ucoef = args.ucoef_burnin + (args.w_unsup - args.ucoef_burnin) * t_u

                loss_total = loss_us + (ucoef * ru) * loss_unsup

                lpl = torch.tensor(0.0, device=device)
                st_pl = {"pl": torch.tensor(0.0, device=device), "pl_ratio": torch.tensor(0.0, device=device)}
                rp = 0.0  # NEW: default
                if args.use_pl and (epoch >= args.pl_start_epoch):
                    lpl, st_pl = pseudo_label_loss(x_a, x_b)

                    rp = rampup_sigmoid((epoch - args.pl_start_epoch) / max(1.0, args.pl_ramp_epochs))
                    loss_total = loss_total + (args.w_pl * rp) * lpl

                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(shape_loop.parameters()), max_norm=0.5)
                optimizer.step()

                if args.use_pl:
                    with torch.no_grad():
                        ema_update(teacher_model, model, args.ema_decay)
                        ema_update(teacher_shape, shape_loop, args.ema_decay)

                # ----- meters -----
                loss_meter["us"] += float(loss_us.detach().cpu())
                loss_meter["unsup"] += float(loss_unsup.detach().cpu())
                loss_meter["inv"] += float(st_u["inv"].cpu())
                loss_meter["cons"] += float(st_u["cons"].cpu())
                loss_meter["prior"] += float(st_u["prior"].cpu())

                loss_meter["nll_mean"] += float(st_u["nll_mean"].cpu())
                loss_meter["logp"] += float(st_u["logp_mean"].cpu())

                loss_meter["geo"] += float(st_u["geo"].cpu())

                loss_meter["cv"] += float(st_u["cv"].cpu())
                loss_meter["conf"] += float(st_u["conf_mean"].cpu())
                loss_meter["lam"] += float(st_u["lam_mean"].cpu())
                loss_meter["pl"] += float(st_pl["pl"].cpu())
                loss_meter["pl_ratio"] += float(st_pl["pl_ratio"].cpu())
                loss_meter["ru"] += float(ru)
                loss_meter["fg"] += float(st_u["fg_mean"].cpu())
                loss_meter["bad_ratio"] += float(st_u["bad_ratio"].cpu())
                loss_meter["rp"] += float(rp)
                loss_meter["area"] += float(st_u["area"].cpu())
                loss_meter["bad_small"] += float(st_u["bad_small_ratio"].cpu())
                loss_meter["bad_big"] += float(st_u["bad_big_ratio"].cpu())
                loss_meter["w_cons"] += float(st_u["w_cons_mean"].cpu())

                step += 1
                pbar.update(1)

        pbar.close()
        scheduler.step()

        # log avg
        denom = max(1, step)

        msg = (
            f"[Epoch {epoch}] "
            f"ru={loss_meter['ru'] / denom:.3f} | "
            f"us={loss_meter['us'] / denom:.4f} | unsup={loss_meter['unsup'] / denom:.4f} "
            f"(inv={loss_meter['inv'] / denom:.4f}, cons={loss_meter['cons'] / denom:.4f}, "
            f"geo={loss_meter['geo'] / denom:.4f}, prior={loss_meter['prior'] / denom:.4f}, "
            f"nll={loss_meter['nll_mean'] / denom:.2f}, conf={loss_meter['conf'] / denom:.3f}, "
            f"lam={loss_meter['lam'] / denom:.4f}, cv={loss_meter['cv'] / denom:.4f}, "
            f"fg={loss_meter['fg'] / denom:.3f}, bad={loss_meter['bad_ratio'] / denom:.3f}, "
            f"area={loss_meter['area'] / denom:.4f}) "
            f"| pl={loss_meter['pl'] / denom:.4f} (ratio={loss_meter['pl_ratio'] / denom:.3f}, rp={loss_meter['rp'] / denom:.3f})"
            f"ucoef={ucoef:.3f} | "
        )

        print(msg)

        def _pick_domain_samples(loader, want_derm: bool, epoch: int, max_tries=200, n=4, fg_range=None):


            try:
                num_batches = len(loader)
            except Exception:
                num_batches = 0

            g = torch.Generator(device="cpu")

            g.manual_seed(int(args.seed + 10000 + epoch * 13 + (1 if want_derm else 0)))

            it_vis = iter(loader)

            if num_batches > 0:
                skip = int(torch.randint(0, num_batches, (1,), generator=g).item())
                for _ in range(skip):
                    try:
                        next(it_vis)
                    except StopIteration:
                        it_vis = iter(loader)
                        break


            imgs, msks = [], []
            got = 0
            tries = 0
            while (got < n) and (tries < max_tries):
                tries += 1
                try:
                    IMG, MSK1ch, _, setseq, _ = next(it_vis)
                    IMG = IMG.cpu()
                    MSK1ch = MSK1ch.cpu()
                except StopIteration:
                    it_vis = iter(loader)
                    continue

                setseq = setseq.long().cpu()
                mask = (setseq == args.derm_id) if want_derm else (setseq != args.derm_id)
                if mask.any():
                    if want_derm and (fg_range is not None):
                        m = MSK1ch[mask]
                        if m.ndim == 4:
                            m = m.squeeze(1)  # [k,H,W]
                        gt = (m.long() > 0).float()    # [k,H,W]
                        fg_ratio = gt.mean(dim=(1, 2))  # [k]
                        keep = (fg_ratio >= fg_range[0]) & (fg_ratio <= fg_range[1])
                        if keep.any():
                            imgs.append(IMG[mask][keep])
                            msks.append(MSK1ch[mask][keep])
                            got += int(keep.sum().item())
                    else:
                        imgs.append(IMG[mask])
                        msks.append(MSK1ch[mask])
                        got += int(mask.sum().item())

            if got == 0:
                return None, None

            IMG = torch.cat(imgs, dim=0)[:n].detach().cpu()
            MSK = torch.cat(msks, dim=0)[:n].detach().cpu()
            return IMG, MSK

        if epoch % 5 == 0:

            img_us, msk_us = _pick_domain_samples(val_loader, want_derm=False, epoch=epoch, max_tries=200, n=4)
            img_d, msk_d = _pick_domain_samples(val_loader, want_derm=True, epoch=epoch, max_tries=200, n=4)

            if (img_us is None) or (img_d is None):
                img_us, msk_us = _pick_domain_samples(train_loader, want_derm=False, epoch=epoch, max_tries=200, n=4)
                img_d, msk_d = _pick_domain_samples(train_loader, want_derm=True,epoch=epoch, max_tries=200, n=4)

            if (img_us is None) or (img_d is None):
                print("[VIS] cannot find both US and DERM for visualization.")
            else:
                vis_dir = os.path.join(args.save_dir, "vis")

                visualize_us_and_derm(epoch, img_us, msk_us, img_d, msk_d, os.path.join(vis_dir, "rand"))

                img_us_fix, msk_us_fix = _pick_domain_samples(val_loader, want_derm=False, epoch=0, n=4)
                img_d_small, msk_d_small = _pick_domain_samples(val_loader, want_derm=True, epoch=0, n=4,
                                                                fg_range=(0.00, 0.25))
                img_d_large, msk_d_large = _pick_domain_samples(val_loader, want_derm=True, epoch=0, n=4,
                                                                fg_range=(0.40, 1.00))

                if (img_us_fix is None) or (img_d_small is None) or (img_d_large is None):
                    img_us_fix, msk_us_fix = _pick_domain_samples(train_loader, want_derm=False, epoch=0, n=4)
                    img_d_small, msk_d_small = _pick_domain_samples(train_loader, want_derm=True, epoch=0, n=4,
                                                                    fg_range=(0.00, 0.25))
                    img_d_large, msk_d_large = _pick_domain_samples(train_loader, want_derm=True, epoch=0, n=4,
                                                                    fg_range=(0.40, 1.00))

                if (img_us_fix is not None) and (img_d_small is not None):
                    visualize_us_and_derm(epoch, img_us_fix, msk_us_fix, img_d_small, msk_d_small,
                                          os.path.join(vis_dir, "fixed_small"))

                if (img_us_fix is not None) and (img_d_large is not None):
                    visualize_us_and_derm(epoch, img_us_fix, msk_us_fix, img_d_large, msk_d_large,
                                          os.path.join(vis_dir, "fixed_large"))



        if (epoch % args.eval_every) == 0:
            metrics = eval_full_val(epoch, save_root=os.path.join(args.save_dir, "val_eval"))


            score = metrics.get("derm_large_0.50", float("nan"))


            if (score == score) and (score > best_score):
                best_score = score
                best_epoch = epoch
                print(f"[BEST] update at epoch={epoch} | score={best_score:.4f} -> save {best_path}")

                ckpt_best = {
                    "model": model.state_dict(),
                    "shape_loop": shape_loop.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_metric": "derm_large@0.50",
                }
                if args.use_pl:
                    ckpt_best["teacher_model"] = teacher_model.state_dict()
                    ckpt_best["teacher_shape"] = teacher_shape.state_dict()

                torch.save(ckpt_best, best_path)


        ckpt_out = {
            "model": model.state_dict(),
            "shape_loop": shape_loop.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
        }
        if args.use_pl:
            ckpt_out["teacher_model"] = teacher_model.state_dict()
            ckpt_out["teacher_shape"] = teacher_shape.state_dict()


        latest_path = Path(args.save_dir) / "stage2_latest.pth"
        torch.save(ckpt_out, latest_path)


        save_every = 5
        if (epoch % save_every == 0) or (epoch == args.max_epoch - 1):
            epoch_path = Path(args.save_dir) / f"stage2_epoch{epoch:03d}.pth"
            torch.save(ckpt_out, epoch_path)


        keep_last = 10
        ckpts = sorted(Path(args.save_dir).glob("stage2_epoch*.pth"))
        if len(ckpts) > keep_last:
            for p in ckpts[:-keep_last]:
                try:
                    p.unlink()
                except Exception:
                    pass

    print("Stage2 done.")

if __name__ == "__main__":
    main()
