# metrics_uda.py
# 评测：Dice / IoU / Boundary-F1@τ / HD95 / AUPRC（macro+micro）/ Brier
# 用法示例：
#   python metrics_uda.py \
#       --pred_dir /path/to/preds \
#       --gt_dir   /path/to/masks \
#       --out_csv  metrics_per_image.csv \
#       --out_json metrics_summary.json \
#       --threshold 0.5 \
#       --taus_px 2 4 8 \
#       --taus_pct 0.01 0.02 0.03 \
#       --spacing_mm 1.0 1.0


from __future__ import annotations
import os, json, math, argparse, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import average_precision_score

SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_NUMPY_EXTS = {".npy"}

# ---------------- I/O ----------------
def imread_any(fp: Path, as_float: bool = True) -> np.ndarray:
    ext = fp.suffix.lower()
    if ext in SUPPORTED_NUMPY_EXTS:
        arr = np.load(str(fp))
        return arr.astype(np.float32, copy=False) if as_float else arr
    elif ext in SUPPORTED_IMG_EXTS:
        img = Image.open(fp).convert("L")  # 灰度
        arr = np.array(img)
        if as_float:
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:  # 0..255 -> 0..1
                arr /= 255.0
        return arr
    else:
        raise ValueError(f"Unsupported file extension: {ext}. File: {fp}")

def _ensure_single_channel(arr: np.ndarray, for_gt: bool) -> np.ndarray:
    """将多通道数组转换为单通道：
       - 二分类概率(2通道)：默认取第1通道作为前景概率
       - 单通道：原样返回
       - 其它情况：尝试 squeeze；仍>2D则报错提示用户导出单通道
    """
    a = arr
    if a.ndim == 2:
        return a
    # 可能是 CHW
    if a.ndim == 3 and a.shape[0] in (1, 2):
        if a.shape[0] == 1:
            return a[0]
        # 2通道：取第1通道为前景
        return a[1]
    # 可能是 HWC
    if a.ndim == 3 and a.shape[2] in (1, 2):
        if a.shape[2] == 1:
            return a[..., 0]
        return a[..., 1]
    # 其它高维情况
    if for_gt:
        raise ValueError("GT应为单通道二值或2通道(含前景通道=1)。请导出单通道 mask 或 2通道前景概率。")
    else:
        raise ValueError("预测应为单通道概率或2通道(前景=通道1)。请导出单通道概率图。")

def pair_paths(pred_dir: Path, gt_dir: Path) -> List[Tuple[Path, Path]]:
    pred_map: Dict[str, Path] = {}
    for p in pred_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (SUPPORTED_IMG_EXTS | SUPPORTED_NUMPY_EXTS):
            pred_map[p.stem] = p
    pairs, missing_gt = [], []
    for g in gt_dir.rglob("*"):
        if g.is_file() and g.suffix.lower() in (SUPPORTED_IMG_EXTS | SUPPORTED_NUMPY_EXTS):
            stem = g.stem
            if stem in pred_map:
                pairs.append((pred_map[stem], g))
            else:
                missing_gt.append(stem)
    if not pairs:
        raise RuntimeError("未找到可配对的文件（按文件名stem匹配）。请确保预测与GT同名。")
    if missing_gt:
        print(f"[warn] {len(missing_gt)} 个GT未匹配到预测。示例: {missing_gt[:5]}")
    return sorted(pairs, key=lambda x: x[0].stem.lower())



# ---------------- 基础指标 ----------------
def binarize_from_prob(prob: np.ndarray, thr: float) -> np.ndarray:
    return (prob >= thr).astype(np.uint8)

def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> Tuple[float, float]:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    pp = pred_bin.sum()
    gg = gt_bin.sum()
    union = pp + gg - inter
    dice = (2.0 * inter + eps) / (pp + gg + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


# ---------------- 边界与BF1 ----------------
def extract_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("只支持2D边界提取。")
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), iterations=1, border_value=0)
    boundary = mask ^ eroded.astype(np.uint8)
    return boundary.astype(np.uint8)

def boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray,
                taus_px: List[float], taus_pct_diag: List[float]) -> Dict[str, float]:
    H, W = gt_bin.shape
    diag = math.sqrt(H*H + W*W)
    pb, gb = extract_boundary(pred_bin), extract_boundary(gt_bin)
    results: Dict[str, float] = {}
    if pb.sum() == 0 and gb.sum() == 0:
        # 双空边界视为完美对齐
        for t in taus_px: results[f"BF1@{int(t)}px"] = 1.0
        for pct in taus_pct_diag: results[f"BF1@{pct*100:.1f}%diag"] = 1.0
        return results

    dt_gb = distance_transform_edt(1 - gb)
    dt_pb = distance_transform_edt(1 - pb)

    def pr_at_tau(tau_px: float) -> Tuple[float, float]:
        if pb.sum() == 0:
            precision = 1.0 if gb.sum() == 0 else 0.0
        else:
            precision = float((dt_gb[pb.astype(bool)] <= tau_px).sum()) / max(1, int(pb.sum()))
        if gb.sum() == 0:
            recall = 1.0 if pb.sum() == 0 else 0.0
        else:
            recall = float((dt_pb[gb.astype(bool)] <= tau_px).sum()) / max(1, int(gb.sum()))
        return precision, recall

    for t in taus_px:
        P, R = pr_at_tau(float(t))
        results[f"BF1@{int(t)}px"] = float((2*P*R) / (P+R+1e-8))
    for pct in taus_pct_diag:
        tau_px = pct * diag
        P, R = pr_at_tau(float(tau_px))
        results[f"BF1@{pct*100:.1f}%diag"] = float((2*P*R) / (P+R+1e-8))
    return results

# ---------------- HD95 ----------------
def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray,
         pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> Optional[float]:
    # 双空：不定义
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return None

    pb = extract_boundary(pred_bin) if pred_bin.sum() > 0 else pred_bin
    gb = extract_boundary(gt_bin) if gt_bin.sum() > 0 else gt_bin

    sy, sx = float(pixel_spacing[0]), float(pixel_spacing[1])
    H, W = gt_bin.shape
    diag_phys = math.sqrt((H*sy)**2 + (W*sx)**2)

    # 若一侧为空，返回对角线物理长度作为大距离（保守、可解释）
    if pb.sum() == 0 and gb.sum() > 0:
        return diag_phys
    if gb.sum() == 0 and pb.sum() > 0:
        return diag_phys

    # 正常计算：有向距离95分位的平均（物理单位）
    dt_gb_phys = distance_transform_edt(1 - gb, sampling=(sy, sx))
    dt_pb_phys = distance_transform_edt(1 - pb, sampling=(sy, sx))
    d_pred_to_gt = dt_gb_phys[pb.astype(bool)]
    d_gt_to_pred = dt_pb_phys[gb.astype(bool)]

    def p95(arr: np.ndarray) -> float:
        return float(np.percentile(arr, 95)) if arr.size else 0.0

    return 0.5 * (p95(d_pred_to_gt) + p95(d_gt_to_pred))

# ---------------- 单图指标 ----------------
def compute_metrics_for_pair(pred_fp: Path, gt_fp: Path, thr: float,
                             taus_px: List[float], taus_pct_diag: List[float],
                             pixel_spacing: Tuple[float, float]) -> Dict[str, float]:
    # 读取
    pred_arr = _ensure_single_channel(imread_any(pred_fp, as_float=True), for_gt=False)
    gt_arr_raw = imread_any(gt_fp, as_float=False)
    gt_arr = _ensure_single_channel(gt_arr_raw, for_gt=True)

    # GT二值化（兼容0/255或0/1）
    if gt_arr.dtype != np.uint8:
        gt_bin = (gt_arr.astype(np.float32) >= 0.5).astype(np.uint8)
    else:
        gt_bin = (gt_arr > 127).astype(np.uint8)

    # 概率裁剪与阈值化
    pred_prob = pred_arr.astype(np.float32)
    if pred_prob.max() > 1.0:  # 兼容0..255
        pred_prob = np.clip(pred_prob / 255.0, 0.0, 1.0)
    else:
        pred_prob = np.clip(pred_prob, 0.0, 1.0)
    pred_bin = binarize_from_prob(pred_prob, thr)

    # Dice / IoU
    dice, iou = dice_iou(pred_bin, gt_bin)

    # BF1
    bf = boundary_f1(pred_bin, gt_bin, taus_px=taus_px, taus_pct_diag=taus_pct_diag)

    # HD95
    hd = hd95(pred_bin, gt_bin, pixel_spacing=pixel_spacing)
    H, W = gt_bin.shape
    sy, sx = pixel_spacing
    diag_phys = math.sqrt((H * sy) ** 2 + (W * sx) ** 2)
    n_hd = hd / (diag_phys + 1e-8)

    if hd is None: hd = float("nan")

    # AUPRC（per-image，用于macro）
    y_true = gt_bin.reshape(-1).astype(np.uint8)
    y_score = pred_prob.reshape(-1).astype(np.float32)
    try:
        ap_img = average_precision_score(y_true, y_score)
    except Exception:
        ap_img = float("nan")

    # Brier
    brier = float(np.mean((y_score - y_true)**2))

    out: Dict[str, float] = {"Dice": dice, "IoU": iou, "HD95": hd,
                             "AUPRC_image": ap_img, "Brier": brier,"nHD95":n_hd}
    out.update(bf)
    return out, y_true, y_score

# ---------------- 汇总 ----------------
def aggregate_results(per_image: List[Dict[str, float]],
                      y_true_all: np.ndarray, y_score_all: np.ndarray) -> Dict[str, float]:
    keys = list(per_image[0].keys())
    mean_dict: Dict[str, float] = {}
    for k in keys:
        vals = np.array([d[k] for d in per_image], dtype=np.float64)
        if k == "HD95":
            mean_dict[k] = float(np.nanmean(vals))
        elif k == "AUPRC_image":
            mean_dict["AUPRC_macro"] = float(np.nanmean(vals))
        else:
            mean_dict[k] = float(np.nanmean(vals))
    # micro-AUPRC
    try:
        ap_micro = average_precision_score(y_true_all, y_score_all)
    except Exception:
        ap_micro = float("nan")
    mean_dict["AUPRC_micro"] = float(ap_micro)
    return mean_dict

# ---------------- 主程序 ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute Dice/IoU/Boundary-F1@τ/HD95/AUPRC(macro+micro)/Brier for 2D segmentation datasets."
    )
    parser.add_argument("--pred_dir", type=str, required=True, help="预测概率/分割图文件夹（PNG/JPG/TIF/BMP或NPY）")
    parser.add_argument("--gt_dir", type=str, required=True, help="GT/MASK文件夹（PNG/JPG/TIF/BMP或NPY）")
    parser.add_argument("--out_csv", type=str, default="metrics_per_image.csv", help="输出逐图CSV")
    parser.add_argument("--out_json", type=str, default="metrics_summary.json", help="输出均值JSON")
    parser.add_argument("--threshold", type=float, default=0.5, help="概率二值化阈值(0..1)")
    parser.add_argument("--taus_px", type=float, nargs="*", default=[2, 4, 8], help="Boundary-F1 像素容差")
    parser.add_argument("--taus_pct", type=float, nargs="*", default=[0.01, 0.02, 0.03], help="Boundary-F1 图像对角线百分比容差")
    parser.add_argument("--spacing_mm", type=float, nargs=2, default=[1.0, 1.0], help="像素间距(mm): sy sx，用于HD95物理单位")
    args = parser.parse_args()

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    pairs = pair_paths(pred_dir, gt_dir)

    per_rows: List[Dict[str, float]] = []
    y_true_buf, y_score_buf = [], []
    csv_fieldnames: Optional[List[str]] = None

    for pred_fp, gt_fp in pairs:
        m, y_t, y_s = compute_metrics_for_pair(
            pred_fp, gt_fp, thr=args.threshold,
            taus_px=args.taus_px, taus_pct_diag=args.taus_pct,
            pixel_spacing=(args.spacing_mm[0], args.spacing_mm[1])
        )
        row = {"name": pred_fp.stem}; row.update(m)
        per_rows.append(row)
        y_true_buf.append(y_t); y_score_buf.append(y_s)
        if csv_fieldnames is None: csv_fieldnames = list(row.keys())

    mean_dict = aggregate_results(
        [{k:v for k,v in r.items() if k!="name"} for r in per_rows],
        np.concatenate(y_true_buf, axis=0),
        np.concatenate(y_score_buf, axis=0)
    )

    # 写CSV（逐图 + MEAN）
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in per_rows: writer.writerow(r)
        mean_row = {"name": "MEAN"}
        for k in csv_fieldnames:
            if k == "name": continue
            if k in mean_dict:
                mean_row[k] = mean_dict[k]
            else:
                vals = [rr[k] for rr in per_rows if k in rr]
                if vals: mean_row[k] = float(np.nanmean(np.array(vals, dtype=np.float64)))
        writer.writerow(mean_row)


    # 写JSON（均值）
    with open(args.out_json, "w") as f:
        json.dump(mean_dict, f, indent=2)

    # 终端打印
    print("=== Dataset Means ===")
    ordered = ["Dice","IoU"] + [k for k in mean_dict.keys() if k.startswith("BF1@")] + \
              ["HD95","nHD95","AUPRC_macro","AUPRC_micro","Brier"]
    for k in ordered:
        if k in mean_dict: print(f"{k}: {mean_dict[k]:.6f}")
    print(f"\nSaved per-image CSV to: {args.out_csv}")
    print(f"Saved summary JSON to: {args.out_json}")

if __name__ == "__main__":
    main()
