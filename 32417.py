#!/usr/bin/env python3
"""
Marker-based Watershed Cell Segmentation (Non-DL)
CS419 Assignment #3 helper script

What it does
------------
Given a microscopy COLOR image, produces an INSTANCE label image (grayscale) where:
- background pixels = 0
- each cell gets a unique integer label: 1..N

Core method: marker-based watershed + distance transform markers.

Usage examples
--------------
# Single image -> instance labels (16-bit PNG/TIFF)
python 32417.py --img path/to/image.tif --out pred_labels.tif

# Also save a visualization overlay with watershed borders
python 32417.py --img image.tif --out pred_labels.tif --save_vis pred_overlay.png

# Batch evaluate (binary pixel-level F1 + IoU) using label directory
python 32417.py --img_dir images/ --label_dir labels/ --out_dir preds/ --eval
- The dataset images/labels are 16-bit TIFF. This script handles uint16 by normalizing to uint8 for processing.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def _ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize uint16/float images to uint8 for OpenCV processing."""
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    mn, mx = float(img_f.min()), float(img_f.max())
    if mx <= mn + 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    img_n = (img_f - mn) / (mx - mn)
    return (img_n * 255.0).astype(np.uint8)

def read_image_any_depth(path: str) -> np.ndarray:
    """Read image (TIFF/PNG/JPG) preserving bit depth and channels."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # OpenCV returns shape (H,W) for grayscale, (H,W,C) for color.
    return img

def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Convert input to 8-bit BGR for watershed."""
    if img.ndim == 2:
        img8 = normalize_to_uint8(img)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    # color
    img8 = normalize_to_uint8(img)
    if img8.shape[2] == 4:
        img8 = img8[:, :, :3]
    # OpenCV reads as BGR already
    return img8

def extract_feature_channel(bgr8: np.ndarray, channel: str) -> np.ndarray:
    """
    Return a single-channel uint8 image used to build the initial binary mask.
    channel options:
      - gray
      - b,g,r
      - hsv_h, hsv_s, hsv_v
      - lab_l, lab_a, lab_b
    """
    ch = channel.lower()
    if ch == "gray":
        return cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)

    if ch in ("b", "g", "r"):
        idx = {"b": 0, "g": 1, "r": 2}[ch]
        return bgr8[:, :, idx]

    if ch.startswith("hsv_"):
        hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
        idx = {"hsv_h": 0, "hsv_s": 1, "hsv_v": 2}[ch]
        return hsv[:, :, idx]

    if ch.startswith("lab_"):
        lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
        idx = {"lab_l": 0, "lab_a": 1, "lab_b": 2}[ch]
        return lab[:, :, idx]

    raise ValueError(f"Unknown channel: {channel}")

def auto_detect_invert(feat: np.ndarray) -> bool:
    """
    Otomatik olarak invert gerekip gerekmediğini tespit eder.
    Eğer görüntüde çoğu piksel koyu ise (siyah arka plan) ve hücreler parlak ise,
    invert=False döner (hücreler parlak olduğu için).
    Eğer görüntüde çoğu piksel parlak ise (beyaz arka plan) ve hücreler koyu ise,
    invert=True döner (hücreler koyu olduğu için).
    """
    # Histogram analizi
    hist = cv2.calcHist([feat], [0], None, [256], [0, 256])
    
    # Koyu pikseller (0-85) ve parlak pikseller (170-255) oranı
    dark_pixels = hist[0:86].sum()
    bright_pixels = hist[170:256].sum()
    total_pixels = feat.size
    
    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels
    
    # Eğer çoğu piksel koyu ise (siyah arka plan), hücreler parlak olmalı -> invert=False
    # Eğer çoğu piksel parlak ise (beyaz arka plan), hücreler koyu olmalı -> invert=True
    if dark_ratio > 0.3 and dark_ratio > bright_ratio:
        return False  # Siyah arka plan, parlak hücreler
    else:
        return True   # Beyaz/parlak arka plan, koyu hücreler (varsayılan)


# -----------------------------
# Metrics
# -----------------------------
def binary_metrics(pred_inst: np.ndarray, gt_inst: np.ndarray) -> Dict[str, float]:
    """
    Pixel-level (binary) metrics:
      - precision, recall, f1 (as %)
      - iou (0..1)
    Uses foreground mask: inst>0.
    """
    pred = (pred_inst > 0)
    gt = (gt_inst > 0)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    iou = tp / (tp + fp + fn + 1e-9)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_percent": float(100.0 * f1),
        "iou": float(iou),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


# -----------------------------
# Segmentation
# -----------------------------
@dataclass
class Params:
    # feature extraction / contrast
    channel: str = "gray"
    invert: bool = True                   # True if cells appear darker than background in chosen channel
    auto_invert: bool = False             # If True, automatically detect invert based on image histogram
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_tile: int = 8                   # tileGridSize=(tile,tile)

    # denoising
    blur_ksize: int = 5                   # Gaussian blur kernel size
    blur_sigma: float = 0.0

    # thresholding
    thresh_method: str = "otsu"           # "otsu" or "adaptive"
    adaptive_block: int = 51              # odd
    adaptive_C: int = 2

    # morphology
    morph_k: int = 3                      # structuring element size
    open_iter: int = 2
    close_iter: int = 0
    sure_bg_dilate_iter: int = 3

    # distance transform markers
    dist_mask_size: int = 5               # for cv2.distanceTransform
    dist_fg_thresh: float = 0.45          # fraction of dist.max()

    # filtering
    min_area: int = 150                   # remove tiny regions after watershed

def marker_based_watershed(bgr8: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      inst_labels: int32 (H,W) labels where 0=background, 1..N cells
      borders: uint8 (H,W) border mask (255 at watershed lines)
    """
    feat = extract_feature_channel(bgr8, p.channel)

    if p.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(p.clahe_clip),
                                tileGridSize=(int(p.clahe_tile), int(p.clahe_tile)))
        feat = clahe.apply(feat)

    k = _ensure_odd(int(p.blur_ksize))
    if k > 1:
        feat_blur = cv2.GaussianBlur(feat, (k, k), float(p.blur_sigma))
    else:
        feat_blur = feat

    # Otomatik invert tespiti (eğer aktifse)
    actual_invert = p.invert
    if p.auto_invert:
        actual_invert = auto_detect_invert(feat_blur)

    # Threshold: produce binary mask where cells = 255, background = 0
    if p.thresh_method.lower() == "otsu":
        if actual_invert:
            _, bw = cv2.threshold(feat_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, bw = cv2.threshold(feat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif p.thresh_method.lower() == "adaptive":
        block = _ensure_odd(int(p.adaptive_block))
        # AdaptiveThreshold outputs 255 for pixels "above" local threshold. Use invert option by flipping types.
        if actual_invert:
            bw = cv2.adaptiveThreshold(feat_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block, int(p.adaptive_C))
        else:
            bw = cv2.adaptiveThreshold(feat_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, int(p.adaptive_C))
    else:
        raise ValueError(f"Unknown thresh_method: {p.thresh_method}")

    k2 = int(p.morph_k)
    k2 = max(1, k2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

    # Remove salt-and-pepper (opening)
    if p.open_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=int(p.open_iter))

    # Optional closing to fill small holes inside cells
    if p.close_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=int(p.close_iter))

    # Sure background via dilation
    sure_bg = cv2.dilate(bw, kernel, iterations=int(p.sure_bg_dilate_iter))

    # Sure foreground via distance transform threshold
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, int(p.dist_mask_size))
    _, sure_fg = cv2.threshold(dist, float(p.dist_fg_thresh) * float(dist.max()), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers from connected components of sure_fg
    n_cc, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                  # make sure background is 1 instead of 0
    markers[unknown == 255] = 0            # unknown is 0

    # Watershed on original image
    markers_ws = cv2.watershed(bgr8.copy(), markers)

    # Convert to instance labels
    inst = np.zeros(markers_ws.shape, dtype=np.int32)
    inst[markers_ws > 1] = markers_ws[markers_ws > 1] - 1   # 1..N (background=0)
    inst[markers_ws == -1] = 0

    # Remove tiny components and relabel 1..K consecutively
    inst = relabel_and_filter(inst, min_area=int(p.min_area))

    borders = (markers_ws == -1).astype(np.uint8) * 255
    return inst, borders

def relabel_and_filter(inst: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Filter small regions and relabel to 1..K."""
    inst = inst.copy()
    if inst.max() == 0:
        return inst

    # Measure areas per label
    labels, counts = np.unique(inst, return_counts=True)
    # labels[0] should be 0
    keep = set([0])
    for lab, cnt in zip(labels, counts):
        if lab == 0:
            continue
        if cnt >= min_area:
            keep.add(int(lab))

    # Zero out removed
    mask_keep = np.isin(inst, list(keep))
    inst[~mask_keep] = 0

    # Relabel consecutive
    kept_labels = [lab for lab in sorted(keep) if lab != 0]
    mapping = {lab: (i + 1) for i, lab in enumerate(kept_labels)}
    out = np.zeros_like(inst, dtype=np.int32)
    for lab, new_lab in mapping.items():
        out[inst == lab] = new_lab
    return out

def overlay_borders(bgr8: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """Return RGB overlay with green borders (thickened)."""
    out = bgr8.copy()
    # Dilate borders to make them thicker and more visible
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    borders_thick = cv2.dilate(borders, kernel, iterations=2)
    out[borders_thick == 255] = (0, 255, 0)  # BGR green
    return out


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--img", type=str, help="Path to a single input image (tif/png/jpg)")
    src.add_argument("--img_dir", type=str, help="Directory of images (batch mode)")

    ap.add_argument("--out", type=str, default=None, help="Output label image path for single-image mode")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory for batch mode")

    ap.add_argument("--label", type=str, default=None, help="Optional ground-truth label image for evaluation")
    ap.add_argument("--label_dir", type=str, default=None, help="GT label directory for batch evaluation")
    ap.add_argument("--eval", action="store_true", help="Compute pixel-level (binary) F1 and IoU when GT is provided")

    ap.add_argument("--save_vis", type=str, default=None, help="Optional path to save border overlay visualization")
    ap.add_argument("--ext", type=str, default=".tif", help="Output extension in batch mode (e.g., .tif or .png)")

    # Parameters
    ap.add_argument("--channel", type=str, default="gray",
                    help="Feature channel: gray|b|g|r|hsv_h|hsv_s|hsv_v|lab_l|lab_a|lab_b")
    ap.add_argument("--invert", action="store_true", help="Invert thresholding (useful when cells are darker)")
    ap.add_argument("--no_invert", dest="invert", action="store_false", help="Do NOT invert thresholding")
    ap.set_defaults(invert=True)
    
    ap.add_argument("--auto_invert", action="store_true", help="Automatically detect invert based on image histogram (overrides --invert/--no_invert)")
    ap.add_argument("--no_auto_invert", dest="auto_invert", action="store_false", help="Disable automatic invert detection")
    ap.set_defaults(auto_invert=True)  # Varsayılan olarak otomatik tespit aktif

    ap.add_argument("--thresh_method", type=str, default="otsu", choices=["otsu", "adaptive"])
    ap.add_argument("--adaptive_block", type=int, default=51)
    ap.add_argument("--adaptive_C", type=int, default=2)

    ap.add_argument("--use_clahe", action="store_true")
    ap.add_argument("--no_clahe", dest="use_clahe", action="store_false")
    ap.set_defaults(use_clahe=True)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_tile", type=int, default=8)

    ap.add_argument("--blur_ksize", type=int, default=5)
    ap.add_argument("--blur_sigma", type=float, default=0.0)

    ap.add_argument("--morph_k", type=int, default=3)
    ap.add_argument("--open_iter", type=int, default=2)
    ap.add_argument("--close_iter", type=int, default=0)
    ap.add_argument("--sure_bg_dilate_iter", type=int, default=3)

    ap.add_argument("--dist_fg_thresh", type=float, default=0.45)
    ap.add_argument("--dist_mask_size", type=int, default=5)

    ap.add_argument("--min_area", type=int, default=150)

    return ap.parse_args()

def build_params(ns: argparse.Namespace) -> Params:
    return Params(
        channel=ns.channel,
        invert=bool(ns.invert),
        auto_invert=bool(ns.auto_invert),
        use_clahe=bool(ns.use_clahe),
        clahe_clip=float(ns.clahe_clip),
        clahe_tile=int(ns.clahe_tile),
        blur_ksize=int(ns.blur_ksize),
        blur_sigma=float(ns.blur_sigma),
        thresh_method=str(ns.thresh_method),
        adaptive_block=int(ns.adaptive_block),
        adaptive_C=int(ns.adaptive_C),
        morph_k=int(ns.morph_k),
        open_iter=int(ns.open_iter),
        close_iter=int(ns.close_iter),
        sure_bg_dilate_iter=int(ns.sure_bg_dilate_iter),
        dist_mask_size=int(ns.dist_mask_size),
        dist_fg_thresh=float(ns.dist_fg_thresh),
        min_area=int(ns.min_area),
    )

def save_label_image(path: str, inst: np.ndarray) -> None:
    """Save instance label image (prefers uint16 when possible)."""
    out_path = str(path)
    max_id = int(inst.max())
    if max_id <= 65535:
        inst_u = inst.astype(np.uint16)
    else:
        # Rare for typical microscopy, but warn and clip to 16-bit.
        print(f"[WARN] Too many instances ({max_id}); clipping to uint16 range.")
        inst_u = np.clip(inst, 0, 65535).astype(np.uint16)

    ok = cv2.imwrite(out_path, inst_u)
    if not ok:
        raise RuntimeError(f"Could not write output image: {out_path}")

def process_one(img_path: str, out_path: str, p: Params, save_vis: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    raw = read_image_any_depth(img_path)
    bgr8 = to_bgr_uint8(raw)
    inst, borders = marker_based_watershed(bgr8, p)
    save_label_image(out_path, inst)

    if save_vis:
        vis = overlay_borders(bgr8, borders)
        cv2.imwrite(save_vis, vis)

    return inst, raw

def main() -> None:
    ns = parse_args()
    p = build_params(ns)

    if ns.img:
        out_path = ns.out
        if out_path is None:
            # default: <imgname>_pred.tif next to input
            stem = Path(ns.img).stem
            out_path = str(Path(ns.img).with_name(stem + "_pred.tif"))
        inst, _ = process_one(ns.img, out_path, p, save_vis=ns.save_vis)
        print(f"[OK] Saved instance labels: {out_path} (instances={int(inst.max())})")

        if ns.eval and ns.label:
            gt = read_image_any_depth(ns.label)
            gt = gt.astype(np.int32)
            m = binary_metrics(inst, gt)
            print("[EVAL] Pixel-level (binary) metrics")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1 (%):    {m['f1_percent']:.2f}")
            print(f"  IoU:       {m['iou']:.4f}")

    else:
        img_dir = Path(ns.img_dir)
        out_dir = Path(ns.out_dir) if ns.out_dir else (img_dir / "preds")
        out_dir.mkdir(parents=True, exist_ok=True)

        label_dir = Path(ns.label_dir) if ns.label_dir else None

        metrics_all: List[Dict[str, float]] = []
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

        for img_path in sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts]):
            out_path = out_dir / (img_path.stem + ns.ext)
            inst, _ = process_one(str(img_path), str(out_path), p, save_vis=None)

            if ns.eval and label_dir is not None:
                # assume same stem in labels folder
                # common patterns: <stem>.tif or <stem>_label.tif etc. Try a few.
                candidates = [
                    label_dir / (img_path.stem + ".tif"),
                    label_dir / (img_path.stem + ".tiff"),
                    label_dir / (img_path.stem + ".png"),
                    label_dir / (img_path.stem + ns.ext),
                ]
                gt_path = next((c for c in candidates if c.exists()), None)
                if gt_path is not None:
                    gt = read_image_any_depth(str(gt_path)).astype(np.int32)
                    m = binary_metrics(inst, gt)
                    m["image"] = img_path.name
                    metrics_all.append(m)

        print(f"[OK] Batch complete. Outputs in: {out_dir}")

        if ns.eval and metrics_all:
            # summarize
            f1s = [m["f1_percent"] for m in metrics_all]
            ious = [m["iou"] for m in metrics_all]
            print("[EVAL] Summary over matched GT files")
            print(f"  N images:  {len(metrics_all)}")
            print(f"  F1 (%):    mean={np.mean(f1s):.2f}, std={np.std(f1s):.2f}, min={np.min(f1s):.2f}, max={np.max(f1s):.2f}")
            print(f"  IoU:       mean={np.mean(ious):.4f}, std={np.std(ious):.4f}, min={np.min(ious):.4f}, max={np.max(ious):.4f}")

if __name__ == "__main__":
    main()
