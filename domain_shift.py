# import random
# import argparse
# from pathlib import Path
#
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
#
# def list_images(root: str):
#     root = Path(root)
#     paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
#     paths.sort()
#     return paths
#
# def open_rgb(path: str):
#     return Image.open(path).convert("RGB")
#
# def center_crop(img: Image.Image, ratio: float = 0.9) -> Image.Image:
#     w, h = img.size
#     nw, nh = int(w * ratio), int(h * ratio)
#     left = (w - nw) // 2
#     top = (h - nh) // 2
#     return img.crop((left, top, left + nw, top + nh))
#
# def rgb_hist(img: Image.Image, bins=32):
#     arr = np.asarray(img).astype(np.uint8)
#     hs = []
#     for c in range(3):
#         h, _ = np.histogram(arr[..., c].ravel(), bins=bins, range=(0, 256))
#         hs.append(h.astype(np.float64))
#     h = np.concatenate(hs, axis=0)
#     h = h / (h.sum() + 1e-12)
#     return h
#
# def average_hist(paths, bins, center_crop_flag=False, crop_ratio=0.9):
#     hist = np.zeros(3 * bins, dtype=np.float64)
#     for p in paths:
#         img = open_rgb(p)
#         if center_crop_flag:
#             img = center_crop(img, crop_ratio)
#         hist += rgb_hist(img, bins=bins)
#     hist /= (hist.sum() + 1e-12)
#     return hist
#
# def sample_paths(paths, max_n, rng):
#     if len(paths) > max_n:
#         return rng.sample(paths, max_n)
#     return paths
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--isic_dir", type=str, required=True)
#     parser.add_argument("--ph2_dir", type=str, required=True)
#     parser.add_argument("--ham_dir", type=str, required=True)
#     parser.add_argument("--out", type=str, default="rgb_hist_shift_three_domains.png")
#     parser.add_argument("--bins", type=int, default=32)
#     parser.add_argument("--max_n", type=int, default=500)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--center_crop", action="store_true")
#     parser.add_argument("--crop_ratio", type=float, default=0.9)
#     args = parser.parse_args()
#
#     rng = random.Random(args.seed)
#
#     isic = sample_paths(list_images(args.isic_dir), args.max_n, rng)
#     ph2 = sample_paths(list_images(args.ph2_dir), args.max_n, rng)
#     ham = sample_paths(list_images(args.ham_dir), args.max_n, rng)
#
#     assert len(isic) > 0, "Empty ISIC folder."
#     assert len(ph2) > 0, "Empty PH2 folder."
#     assert len(ham) > 0, "Empty HAM10000 folder."
#
#     isic_hist = average_hist(isic, args.bins, args.center_crop, args.crop_ratio)
#     ph2_hist = average_hist(ph2, args.bins, args.center_crop, args.crop_ratio)
#     ham_hist = average_hist(ham, args.bins, args.center_crop, args.crop_ratio)
#
#     x = np.arange(3 * args.bins)
#
#     plt.figure(figsize=(7.2, 3.2))
#     plt.plot(x, isic_hist, label="ISIC", linewidth=1.8)
#     plt.plot(x, ph2_hist, label="PH2", linewidth=1.8)
#     plt.plot(x, ham_hist, label="HAM10000", linewidth=1.8)
#
#     plt.xlabel("RGB histogram bin")
#     plt.ylabel("Normalized frequency")
#     plt.legend(frameon=False)
#     plt.tight_layout()
#     plt.savefig(args.out, dpi=300)
#     plt.close()
#
#     print(f"[OK] saved to: {args.out}")
#     print(f"ISIC: {len(isic)} images, PH2: {len(ph2)} images, HAM10000: {len(ham)} images")
#
# if __name__ == "__main__":
#     main()



#T步数

import matplotlib.pyplot as plt

# x 轴：TTA steps
K = [0, 1, 5, 10, 20]

# 你的结果
dice_isic = [0.8918, 0.9007, 0.9143, 0.9251, 0.9133]

plt.plot(K, dice_isic, marker='o', linewidth=2)
# 在每个数据点上方标注数值
for x, y in zip(K, dice_isic):
    plt.annotate(
        f'{y:.4f}',          # 保留 4 位小数
        xy=(x, y),           # 数据点位置
        xytext=(0, 8),       # 文本向上偏移 8 points
        textcoords='offset points',
        ha='center',         # 水平居中
        va='bottom',
        fontsize=9
    )

# 在 x=10 位置加一条竖直蓝色虚线
plt.axvline(x=10,  color='lightskyblue', linestyle='--', linewidth=1.5)

plt.xlabel('TTA steps (T)')
plt.ylabel('Dice')
plt.xticks(K)
plt.ylim(0.8, 0.95)  # 你也可以按需要改
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('dice_vs_steps.pdf')  # 论文用（矢量）
plt.savefig('dice_vs_steps.png')  # 预览用
print('Saved: dice_vs_steps.pdf, dice_vs_steps.png')



#重要参数敏感性分析

import os
import matplotlib.pyplot as plt

# ============================================================
# 输出目录
# ============================================================
save_dir = "./sensitivity_figures"
os.makedirs(save_dir, exist_ok=True)


def plot_numeric_sensitivity(
    x,
    dice,
    xlabel,
    save_name,
    default_x,
    xticks=None,
):
    """
    绘制数值型横坐标的超参数敏感性曲线。

    Parameters
    ----------
    x : list[float]
        横坐标参数值。
    dice : list[float]
        对应Dice结果。
    xlabel : str
        横坐标名称。
    save_name : str
        保存文件名。
    default_x : float
        默认参数值。
    xticks : list[float] | None
        横坐标刻度。
    """
    if len(x) != len(dice):
        raise ValueError("x和dice的长度必须一致。")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(
        x,
        dice,
        marker="o",
        linewidth=2,
        markersize=7,
    )

    # 标记默认配置
    ax.axvline(
        default_x,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    # 标注每个点的数值
    y_range = max(dice) - min(dice)
    text_offset = max(0.0008, y_range * 0.08)

    for x_i, y_i in zip(x, dice):
        ax.text(
            x_i,
            y_i + text_offset,
            f"{y_i:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if xticks is not None:
        ax.set_xticks(xticks)

    # 根据数据自动设置纵轴范围
    ax.set_ylim(0.80, 0.95)
    ax.set_yticks([0.80, 0.84, 0.88, 0.92])

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Dice", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, save_name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_categorical_sensitivity(
    labels,
    dice,
    xlabel,
    save_name,
    default_index,
):
    """
    绘制字符串型横坐标的超参数敏感性曲线，
    适用于(delta_start, delta_end)这样的成对参数。
    """
    if len(labels) != len(dice):
        raise ValueError("labels和dice的长度必须一致。")

    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    ax.plot(
        positions,
        dice,
        marker="o",
        linewidth=2,
        markersize=7,
    )

    ax.axvline(
        default_index,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    y_range = max(dice) - min(dice)
    text_offset = max(0.0008, y_range * 0.08)

    for x_i, y_i in zip(positions, dice):
        ax.text(
            x_i,
            y_i + text_offset,
            f"{y_i:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    ax.set_ylim(0.80, 0.95)
    ax.set_yticks([0.80, 0.84, 0.88, 0.92])

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Dice", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, save_name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 1. Closure-consistency weight: w_cycle
# 实验完成后，只替换dice_w_cycle
# ============================================================
w_cycle = [0.050, 0.075, 0.100, 0.125, 0.150]

dice_w_cycle = [
    0.9187,
    0.9228,
    0.9251,
    0.9239,
    0.9205,
]

plot_numeric_sensitivity(
    x=w_cycle,
    dice=dice_w_cycle,
    xlabel=r"Closure-consistency weight $\omega_{\mathrm{cyc}}$",
    save_name="sensitivity_w_cycle.png",
    default_x=0.100,
    xticks=w_cycle,
)


# ============================================================
# 2. Trust-region weight: w_tr
# 实验完成后，只替换dice_w_tr
# ============================================================
w_tr = [0.0250, 0.0375, 0.0500, 0.0625, 0.0750]

dice_w_tr = [
    0.9192,
    0.9231,
    0.9251,
    0.9237,
    0.9209,
]

plot_numeric_sensitivity(
    x=w_tr,
    dice=dice_w_tr,
    xlabel=r"Trust-region weight $\omega_{\mathrm{tr}}$",
    save_name="sensitivity_w_tr.png",
    default_x=0.0500,
    xticks=w_tr,
)


# ============================================================
# 3. Shape-prior strength multiplier: s_p
# 对lambda0和lambda_min同时缩放
# 实验完成后，只替换dice_prior
# ============================================================
prior_multiplier = [0.50, 0.75, 1.00, 1.25, 1.50]

dice_prior = [
    0.9217,
    0.9240,
    0.9251,
    0.9245,
    0.9226,
]

plot_numeric_sensitivity(
    x=prior_multiplier,
    dice=dice_prior,
    xlabel=r"Shape-prior strength multiplier $s_p$",
    save_name="sensitivity_prior_strength.png",
    default_x=1.00,
    xticks=prior_multiplier,
)


# ============================================================
# 4. Confidence-threshold schedule
# 标签格式：delta_start → delta_end
# 实验完成后，只替换dice_confidence
# ============================================================
confidence_labels = [
    "0.93→0.89",
    "0.95→0.91",
    "0.97→0.93",
    "0.98→0.94",
    "0.99→0.95",
]

dice_confidence = [
    0.9139,
    0.9204,
    0.9251,
    0.9190,
    0.9078,
]

plot_categorical_sensitivity(
    labels=confidence_labels,
    dice=dice_confidence,
    xlabel=r"Confidence-threshold schedule "
           r"$(\delta_{\mathrm{start}}\rightarrow\delta_{\mathrm{end}})$",
    save_name="sensitivity_confidence_threshold.png",
    default_index=2,
)


# ============================================================
# 5. Selected-pixel-ratio multiplier: s_r
# 对pl_max_ratio_fg和pl_max_ratio_bg同时缩放
# 实验完成后，只替换dice_pixel_ratio
# ============================================================
pixel_ratio_multiplier = [0.50, 0.75, 1.00, 1.25, 1.50]

dice_pixel_ratio = [
    0.9188,
    0.9229,
    0.9251,
    0.9238,
    0.9207,
]

plot_numeric_sensitivity(
    x=pixel_ratio_multiplier,
    dice=dice_pixel_ratio,
    xlabel=r"Selected-pixel-ratio multiplier $s_r$",
    save_name="sensitivity_pixel_ratio.png",
    default_x=1.00,
    xticks=pixel_ratio_multiplier,
)

print(f"Five sensitivity figures have been saved to: {save_dir}")
