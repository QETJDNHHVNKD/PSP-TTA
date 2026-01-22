
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset.mutlidomain_baseloader import baseloader
from  model.Model import ASF
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from shape_loop import ShapeClosedLoop, soft_dice_loss

import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@torch.no_grad()
def val_shape_loop(model, shape_loop, val_loader, device, tumor_classes=(1,2), eval_derm=False):
    model.eval()
    shape_loop.eval()
    dices = []

    for IMG, MSK1ch, MSKmul, setseq, _ in tqdm(val_loader, desc="Val (ShapeLoop)"):
        mask = (setseq == 3) if eval_derm else (setseq != 3)
        if mask.sum() == 0:
            continue

        IMG = IMG[mask].to(device)
        MSK1ch = MSK1ch[mask].to(device).long().squeeze(1)  # [B,H,W]

        # 和训练一致：二值前景
        tc = torch.tensor(tumor_classes, device=MSK1ch.device)
        MSK1ch_bin = (torch.isin(MSK1ch, tc)).long()
        gt = F.one_hot(MSK1ch_bin, num_classes=2).permute(0,3,1,2).float()[:,1:2]  # [B,1,H,W]

        feats = model.forward_features(IMG)
        _, m1, _, _ = shape_loop(feats)  # [B,1,h,w]

        if gt.shape[-2:] != m1.shape[-2:]:
            gt = F.interpolate(gt, size=m1.shape[-2:], mode="nearest")

        inter = (m1 * gt).sum((1,2,3)) * 2.0
        denom = m1.sum((1,2,3)) + gt.sum((1,2,3)) + 1e-6
        dice = (inter / denom)
        dices.extend(dice.detach().cpu().tolist())

    return float(torch.tensor(dices).mean()) if dices else 0.0


@torch.no_grad()
def vis_stage1_once(model, shape_loop, loader, device, save_path,
                    tumor_classes=(1,2), eval_derm=False, max_items=4):
    model.eval()
    shape_loop.eval()

    # 取一个 batch
    for IMG, MSK1ch, MSKmul, setseq, _ in loader:
        mask = (setseq == 3) if eval_derm else (setseq != 3)
        if mask.sum() == 0:
            continue

        IMG = IMG[mask].to(device)
        MSK1ch = MSK1ch[mask].to(device).long().squeeze(1)  # [B,H,W]

        tc = torch.tensor(tumor_classes, device=device)
        gt = torch.isin(MSK1ch, tc).float().unsqueeze(1)     # [B,1,H,W]

        feats = model.forward_features(IMG)
        z1, m1, z2, m2 = shape_loop(feats)                   # [B,1,H,W]

        # 对齐尺寸
        if gt.shape[-2:] != m1.shape[-2:]:
            gt = F.interpolate(gt, size=m1.shape[-2:], mode="nearest")


        z_gt = shape_loop.mask_encoder(gt)
        m_gt = shape_loop.renderer(z_gt)

        def dice_soft(a, b):
            inter = (a * b).sum((1,2,3)) * 2.0
            denom = a.sum((1,2,3)) + b.sum((1,2,3)) + 1e-6
            return (inter / denom)

        def dice_hard(prob, gt, thr=0.5):
            pred = (prob > thr).float()
            inter = (pred * gt).sum((1, 2, 3)) * 2.0
            denom = pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) + 1e-6
            return inter / denom

        d_mgt_gt_h = dice_hard(m_gt, gt).mean().item()
        d_m1_gt_h = dice_hard(m1, gt).mean().item()

        print(f"[VIS] hard_dice(m1,gt)={d_m1_gt_h:.4f} | hard_dice(m_gt,gt)={d_mgt_gt_h:.4f}")

        d_m1_gt = dice_soft(m1, gt).mean().item()
        d_m2_m1 = dice_soft(m2, m1).mean().item()
        d_mgt_gt = dice_soft(m_gt, gt).mean().item()
        z_cons = F.mse_loss(z1, z_gt).item()


        p, cx, cy, r0, a, b = shape_loop.renderer.decode(z1)
        print(f"[VIS] dice(m1,gt)={d_m1_gt:.4f} | dice(m2,m1)={d_m2_m1:.4f} | "
              f"dice(m_gt,gt)={d_mgt_gt:.4f} | mse(z1,z_gt)={z_cons:.6f}")
        print(f"[VIS] p in [{p.min().item():.3f},{p.max().item():.3f}] | "
              f"cx/cy in [{cx.min().item():.3f},{cx.max().item():.3f}] / [{cy.min().item():.3f},{cy.max().item():.3f}] | "
              f"r0 in [{r0.min().item():.3f},{r0.max().item():.3f}]")


        B = min(max_items, IMG.shape[0])
        fig, axes = plt.subplots(B, 5, figsize=(14, 3*B))
        if B == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(B):
            img = IMG[i].detach().cpu()
            if img.dim() == 3:  # [C,H,W]
                if img.shape[0] > 1:
                    img_show = img.mean(0)
                else:
                    img_show = img[0]
            else:
                img_show = img


            img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)

            gt_i  = gt[i,0].detach().cpu()
            m1_i  = m1[i,0].detach().cpu()
            m2_i  = m2[i,0].detach().cpu()
            mgt_i = m_gt[i,0].detach().cpu()

            axes[i,0].imshow(img_show, cmap="gray"); axes[i,0].set_title("IMG"); axes[i,0].axis("off")
            axes[i,1].imshow(gt_i, cmap="gray");     axes[i,1].set_title("GT");  axes[i,1].axis("off")
            axes[i,2].imshow(m1_i, cmap="gray");     axes[i,2].set_title("m1");  axes[i,2].axis("off")
            axes[i,3].imshow(m2_i, cmap="gray");     axes[i,3].set_title("m2");  axes[i,3].axis("off")
            axes[i,4].imshow(mgt_i, cmap="gray");    axes[i,4].set_title("GT->E->R"); axes[i,4].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[VIS] saved: {save_path}")
        return


@torch.no_grad()
def vis_stage1_parametric(model, shape_loop, loader, device, save_path,
                          tumor_classes=(1,2), eval_derm=False, max_items=3,
                          n_theta=512):

    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    import torch.nn.functional as F

    model.eval()
    shape_loop.eval()

    def z_to_curve(z, H, W):

        p, cx, cy, r0, a, b = shape_loop.renderer.decode(z)  # torch tensors
        # build theta
        theta = torch.linspace(0, 2*np.pi, n_theta, device=z.device, dtype=z.dtype)[None, :]  # [1,T]
        k = torch.arange(1, shape_loop.renderer.K + 1, device=z.device, dtype=z.dtype)[None, :, None]  # [1,K,1]

        # [B,1,T]
        cos_k = torch.cos(k * theta[:, None, :])
        sin_k = torch.sin(k * theta[:, None, :])


        a_ = a[:, :, None]
        b_ = b[:, :, None]


        r = r0[:, None, :] + (a_ * cos_k + b_ * sin_k).sum(dim=1, keepdim=True)


        r = torch.clamp(r, min=1e-3, max=shape_loop.renderer.r_max)


        x = cx[:, None, None] + r * torch.cos(theta[:, None, :])
        y = cy[:, None, None] + r * torch.sin(theta[:, None, :])


        x_pix = (x + 1) * 0.5 * (W - 1)
        y_pix = (y + 1) * 0.5 * (H - 1)

        # ✅ 强制变成 [B, T]
        x_pix = x_pix.reshape(x_pix.shape[0], -1)
        y_pix = y_pix.reshape(y_pix.shape[0], -1)

        r = r.reshape(r.shape[0], -1)  # r(theta) 也同理，防止出现 [B,1,T]


        amp = torch.sqrt(a*a + b*b)

        return {
            "theta": theta.squeeze(0).detach().cpu().numpy(),   # [T]
            "r": r.detach().cpu().numpy(),                      # [B,T]
            "x_pix": x_pix.detach().cpu().numpy(),              # [B,T]
            "y_pix": y_pix.detach().cpu().numpy(),              # [B,T]
            "p": p.detach().cpu().numpy().squeeze(1),          # [B]
            "cx": cx.detach().cpu().numpy().squeeze(1),
            "cy": cy.detach().cpu().numpy().squeeze(1),
            "r0": r0.detach().cpu().numpy().squeeze(1),
            "amp": amp.detach().cpu().numpy(),                 # [B,K]
        }


    for IMG, MSK1ch, MSKmul, setseq, _ in loader:
        mask = (setseq == 3) if eval_derm else (setseq != 3)
        if mask.sum() == 0:
            continue

        IMG = IMG[mask].to(device)
        MSK1ch = MSK1ch[mask].to(device).long().squeeze(1)  # [B,H,W]

        tc = torch.tensor(tumor_classes, device=device)
        gt = torch.isin(MSK1ch, tc).float().unsqueeze(1)     # [B,1,H,W]

        feats = model.forward_features(IMG)
        z1, m1, z2, m2 = shape_loop(feats)

        if gt.shape[-2:] != m1.shape[-2:]:
            gt = F.interpolate(gt, size=m1.shape[-2:], mode="nearest")

        z_gt = shape_loop.mask_encoder(gt)
        m_gt = shape_loop.renderer(z_gt)

        B = min(max_items, gt.shape[0])
        H, W = gt.shape[-2], gt.shape[-1]

        c1 = z_to_curve(z1[:B], H, W)
        c2 = z_to_curve(z2[:B], H, W)
        cgt = z_to_curve(z_gt[:B], H, W)

        K = shape_loop.renderer.K

        from pathlib import Path
        save_path = Path(save_path)
        out_dir = save_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # 文件名基底：epoch_0220_param
        stem = save_path.stem
        suffix = save_path.suffix  # .png

        for i in range(B):

            fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2))

            gt_i = gt[i, 0].detach().cpu().numpy()
            m1_i = m1[i, 0].detach().cpu().numpy()
            m2_i = m2[i, 0].detach().cpu().numpy()
            mgt_i = m_gt[i, 0].detach().cpu().numpy()


            ax.imshow(gt_i, cmap="gray")

            cs1 = ax.contour(m1_i, levels=[0.5], colors=["tab:blue"], linewidths=3)
            cs2 = ax.contour(m2_i, levels=[0.5], colors=["tab:orange"], linewidths=3)
            cs3 = ax.contour(mgt_i, levels=[0.5], colors=["tab:green"], linewidths=3)



            h1 = cs1.collections[0]
            h2 = cs2.collections[0]
            h3 = cs3.collections[0]
            ax.legend([h1, h2, h3],
                      ["z1/SPL->R (m1)", "m1->E->R (m2)", "GT->E->R (m_gt)"],
                      fontsize=8, loc="lower right")

            ax.set_title(f"Contour Overlay (p={c1['p'][i]:.2f}, r0={c1['r0'][i]:.2f})")
            ax.axis("off")

            path_contour = out_dir / f"{stem}_s{i:02d}_contour{suffix}"
            plt.tight_layout()
            plt.savefig(str(path_contour), dpi=200)
            plt.close(fig)
            print(f"[VIS-PARAM] saved: {path_contour}")


            fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
            ax.plot(c1["theta"], c1["r"][i], label="z1")
            ax.plot(c2["theta"], c2["r"][i], label="z2")
            ax.plot(cgt["theta"], cgt["r"][i], label="z_gt")

            ax.set_title("Polar Radius Curve r(θ)")
            ax.set_xlabel("θ")
            ax.set_ylabel("r")
            ax.legend(fontsize=9)

            path_curve = out_dir / f"{stem}_s{i:02d}_curve{suffix}"
            plt.tight_layout()
            plt.savefig(str(path_curve), dpi=200)
            plt.close(fig)
            print(f"[VIS-PARAM] saved: {path_curve}")

            fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
            x = np.arange(1, K + 1)
            ax.bar(x - 0.2, c1["amp"][i], width=0.2, label="z1")
            ax.bar(x, c2["amp"][i], width=0.2, label="z2")
            ax.bar(x + 0.2, cgt["amp"][i], width=0.2, label="z_gt")

            ax.set_title("Fourier Amplitude Spectrum")
            ax.set_xlabel("k")
            ax.set_ylabel("sqrt(a_k^2 + b_k^2)")
            ax.legend(fontsize=9)

            path_spec = out_dir / f"{stem}_s{i:02d}_spec{suffix}"
            plt.tight_layout()
            plt.savefig(str(path_spec), dpi=200)
            plt.close(fig)
            print(f"[VIS-PARAM] saved: {path_spec}")

        return


def build_optimizer(model, adaptive_module, args):
    base_lr = args.lr
    wd = args.weight_decay
    used_ids = set()
    params = []

    for i in range(5):
        group = list(getattr(model, f"layer{i}").parameters())
        used_ids.update(map(id, group))
        params.append({"params": group, "lr": base_lr * 0.5, "weight_decay": wd * 0.5})

    bicst_params = [p for n, p in model.named_parameters() if "bicst" in n and id(p) not in used_ids]
    used_ids.update(map(id, bicst_params))
    params.append({"params": bicst_params, "lr": base_lr, "weight_decay": wd})

    prompt_gen_params, semantic_aligner_params = [], []
    for name, param in model.prompt_generator.named_parameters():
        if id(param) in used_ids:
            continue
        if "semantic_aligner" in name:
            semantic_aligner_params.append(param)
        else:
            prompt_gen_params.append(param)
        used_ids.add(id(param))

    params.append({"params": prompt_gen_params, "lr": base_lr, "weight_decay": wd * 0.1})
    params.append({"params": semantic_aligner_params, "lr": base_lr * 0.1, "weight_decay": wd * 0.05})


    params.append({"params": adaptive_module.prompt_refiner.parameters(), "lr": base_lr * 0.2, "weight_decay": wd})
    params.append({"params": adaptive_module.anomaly_fuser.parameters(), "lr": base_lr * 0.2, "weight_decay": wd})
    params.append({"params": adaptive_module.threshold_module.parameters(), "lr": base_lr, "weight_decay": wd})


    other_params = [p for p in model.parameters() if id(p) not in used_ids]
    params.append({"params": other_params, "lr": base_lr, "weight_decay": wd})

    return torch.optim.AdamW(params)


def _set_bn_eval(m: nn.Module):
    import torch.nn as nn
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):

        m.eval()
        m.track_running_stats = True
        m.momentum = 0.0
        for p in m.parameters():
            p.requires_grad = False

def freeze_module_params(module, freeze=True):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = (not freeze)

def bn_eval_all(*modules):
    for m in modules:
        if m is not None:
            m.apply(_set_bn_eval)

def setup_freeze_for_epoch(model, adaptive_module, epoch, seg_warmup):

    bn_eval_all(
        model,
        getattr(model, "_backbone_decoder", None),
        getattr(adaptive_module, "ema_teacher", None),
        getattr(adaptive_module, "anomaly_detector", None),
        getattr(adaptive_module, "anomaly_fuser", None),
    )

    # 2) 模块冻结/解冻
    if epoch < seg_warmup:
        freeze_module_params(getattr(adaptive_module, "prompt_refiner", None), True)
        freeze_module_params(getattr(adaptive_module, "anomaly_fuser", None), True)
    else:
        freeze_module_params(getattr(adaptive_module, "prompt_refiner", None), False)
        freeze_module_params(getattr(adaptive_module, "anomaly_fuser", None), False)

        for name, module in model.named_modules():
            if "layer0" in name:
                freeze_module_params(module, True)

def enter_target_phase_once(adaptive_module):

    ema_t = getattr(adaptive_module, "ema_teacher", None)
    if ema_t is not None:
        freeze_module_params(ema_t, True)

        try:
            from torch import nn as _nn
            def _apply_bn_eval_safe(_m):
                if _m is not None:
                    _m.apply(_set_bn_eval)

            _apply_bn_eval_safe(getattr(ema_t, None if ema_t is None else "__class__", None) and ema_t)  # 仅为易读性
            _apply_bn_eval_safe(getattr(adaptive_module, "anomaly_detector", None))
            _apply_bn_eval_safe(getattr(adaptive_module, "anomaly_fuser", None))
        except Exception:
            pass
        ema_t.eval()



def loss_schedule(epoch, seg_dc, pri_cl, align_loss, distill=0.0, prompt_reg=0.0, is_source=True):
    if is_source:
        if epoch < 10:
            return seg_dc * 1.0 + align_loss * 0.005
        elif epoch < 15:
            return seg_dc * 0.9 + pri_cl * 0.05 + align_loss * 0.01
        else:
            return seg_dc * 0.8 + pri_cl * 0.01 + align_loss * 0.1
    else:

        return seg_dc * 0.8 + pri_cl * 0.01 + align_loss * 0.1 + distill * 0.05 + prompt_reg * 0.08



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=r'..\Dataset')
    parser.add_argument("--data_configuration",default=r'..\dataset_config.yaml')
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epoch", default=501, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--warmup_epoch", default=20, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--log_flag", action="store_true")
    parser.add_argument("--log_name", default="MOFO")
    parser.add_argument("--save_model_dir", default="output")


    parser.add_argument("--resume_path",  type=str, help="Path to checkpoint to resume from")

    parser.add_argument('--prompt_noise', action='store_true')
    parser.add_argument('--prompt_noise_std', type=float, default=0.02)

    args = parser.parse_args()

    Path(args.save_model_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = baseloader(args)

    prompt_generator = nn.Identity()

    model = ASF(class_num=2, task_prompt="word_embedding", prompt_generator=prompt_generator, use_anomaly_detection=False).to(device)

    model.disable_decoder_dropout = True

    shape_loop = ShapeClosedLoop(out_hw=args.input_size, K=8, sharpness=40.0).to(device)


    train_params = [p for p in model.parameters() if p.requires_grad] + list(shape_loop.parameters())

    if hasattr(model, "organ_embedding"):
        model.organ_embedding.requires_grad_(False)
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)


    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
    writer = SummaryWriter(f"output/{args.log_name}/log") if args.log_flag else None

    best_dice = 0.0

    start_epoch = 0
    if args.resume_path is not None and os.path.exists(args.resume_path):
        print(f"🧩 Resume from checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)

        if "shape_loop" in checkpoint:
            shape_loop.load_state_dict(checkpoint["shape_loop"], strict=False)

        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])


        best_dice = checkpoint.get('best_dice', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1


    for epoch in range(start_epoch, args.max_epoch):

        warmup = 10
        ramp = 50
        s0, s1 = 40.0, 120.0

        if epoch < warmup:
            shape_loop.renderer.sharpness = s0
        elif epoch < warmup + ramp:
            tt = (epoch - warmup) / float(ramp)
            shape_loop.renderer.sharpness = s0 + (s1 - s0) * tt
        else:
            shape_loop.renderer.sharpness = s1


        epoch_losses = {k: 0.0 for k in ["seg", "cons", "rec", "gt_rec"]}
        num_samples = 0
        tc = torch.tensor([1, 2], device=device)

        model.train()
        shape_loop.train()
        for IMG, MSK1ch, MSKmul, setseq, _ in tqdm(train_loader, desc=f"Train {epoch}"):

            #源域训练
            mask = setseq != 3
            if mask.sum() == 0:
                continue

            IMG = IMG[mask].to(device, non_blocking=True)
            MSK1ch = MSK1ch[mask].to(device, non_blocking=True).long().squeeze(1)  # [B,H,W]


            bs = IMG.shape[0]
            num_samples += bs


            MSK1ch_bin = torch.isin(MSK1ch, tc).long()  # [B,H,W]

            gt = MSK1ch_bin.float().unsqueeze(1)  # [B,1,H,W]

            feats = model.forward_features(IMG)

            z1, m1, z2, m2 = shape_loop(feats)


            if gt.shape[-2:] != m1.shape[-2:]:
                gt = F.interpolate(gt, size=m1.shape[-2:], mode="nearest")


            term_seg = F.binary_cross_entropy(m1.clamp(1e-6, 1 - 1e-6), gt) + soft_dice_loss(m1, gt)


            z_gt = shape_loop.mask_encoder(gt)
            m_gt = shape_loop.renderer(z_gt)
            term_gt_rec = F.binary_cross_entropy(m_gt.clamp(1e-6, 1 - 1e-6), gt) + soft_dice_loss(m_gt, gt)


            term_cons = F.mse_loss(z1, z_gt.detach())


            m1_det = (m1.detach() > 0.5).float()
            term_rec = F.binary_cross_entropy(m2.clamp(1e-6, 1 - 1e-6), m1_det) + soft_dice_loss(m2, m1_det)


            loss = term_seg + 0.1 * term_cons + 0.5 * term_rec + 1.0 * term_gt_rec

            optimizer.zero_grad()
            loss.backward()

            total_params = []
            for g in optimizer.param_groups:
                total_params += list(g["params"])
            torch.nn.utils.clip_grad_norm_(total_params, max_norm=0.5)

            optimizer.step()

            epoch_losses["seg"] += term_seg.item() * bs
            epoch_losses["cons"] += term_cons.item() * bs
            epoch_losses["rec"] += term_rec.item() * bs
            epoch_losses["gt_rec"] += term_gt_rec.item() * bs

        den = max(1, num_samples)
        for k in epoch_losses:
            epoch_losses[k] /= den
        print("Losses → " + ", ".join([f"{k}:{v:.4f}" for k, v in epoch_losses.items()]))

        checkpoint = {
            "model": model.state_dict(),
            "shape_loop": shape_loop.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
        }


        torch.save(checkpoint, Path(args.save_model_dir) / f"checkpoint_latest.pth")


        avg_dice = val_shape_loop(model, shape_loop, val_loader, device, tumor_classes=(1, 2), eval_derm=False)

        if epoch % 10 == 0 or epoch == args.max_epoch - 1:
            vis_dir = Path(args.save_model_dir) / "vis_stage1"
            vis_dir.mkdir(parents=True, exist_ok=True)

            vis_stage1_once(                            #可视化拼图
                model, shape_loop, val_loader, device,
                save_path=str(vis_dir / f"epoch_{epoch:04d}.png"),
                tumor_classes=(1, 2), eval_derm=False, max_items=4
            )

            vis_stage1_parametric(                           #可视化曲线
                model, shape_loop, val_loader, device,
                save_path=str(vis_dir / f"epoch_{epoch:04d}_param.png"),
                tumor_classes=(1, 2), eval_derm=False, max_items=3
            )

        print(f" 🔍 Val Dice (source): {avg_dice:.4f}")

        # ✅ 只保存best模型
        if epoch % 5 == 0:
            torch.save(model.state_dict(), Path(args.save_model_dir) / f"model_epoch_{epoch}.pth")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'shape_loop': shape_loop.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
            }, Path(args.save_model_dir) / "best_model.pth")
            print(f" 💾 Save best model. Dice={best_dice:.4f}")

        if epoch % 10 == 0:
            print(f"  📈 Best Dice so far: {best_dice:.4f}")

        if writer:
            writer.add_scalar("train/seg", epoch_losses["seg"], epoch)
            writer.add_scalar("train/cons", epoch_losses["cons"], epoch)
            writer.add_scalar("train/rec", epoch_losses["rec"], epoch)
            writer.add_scalar("train/gt_rec", epoch_losses["gt_rec"], epoch)
            writer.add_scalar("valid/dice_source", avg_dice, epoch)

        scheduler.step()

if __name__ == "__main__":
    main()












