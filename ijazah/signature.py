"""Deteksi tanda tangan: thresholding -> morphology -> analisis komponen terhubung."""
import cv2
import numpy as np
from .config import SIG_MIN_INK_RATIO, SIG_MIN_WIDTH_RATIO, SIG_MIN_COMPONENT_AREA


def detect_signature(gray_roi: np.ndarray, return_debug: bool = False):
    g = gray_roi
    # 1) Normalisasi latar (hilangkan guilloche/kertas kekuningan) + blur ringan
    k = max(31, (min(g.shape) // 2) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(g, cv2.GaussianBlur(bg, (0, 0), k / 3), scale=255)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)

    # 2) Thresholding: Otsu dibatasi agar ROI kosong tidak "menyala"
    otsu_t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(otsu_t, 200, float(np.median(norm)) - 40)
    binary = (norm < t).astype(np.uint8) * 255

    # 3) Morphology: opening (buang noise) lalu closing (sambung goresan)
    se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se_o)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, se_c)

    # 4) Analisis komponen terhubung
    n, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    H, W = clean.shape
    comps = [(stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH])
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= SIG_MIN_COMPONENT_AREA]
    ink_ratio = float((clean > 0).sum()) / (H * W)
    max_w_ratio = max((c[1] for c in comps), default=0) / W
    present = bool(comps) and ink_ratio >= SIG_MIN_INK_RATIO and max_w_ratio >= SIG_MIN_WIDTH_RATIO

    info = {"present": present, "ink_ratio": round(ink_ratio, 4),
            "max_width_ratio": round(float(max_w_ratio), 3), "n_components": len(comps)}
    if return_debug:
        return info, {"normalized": norm, "binary": binary, "morphology": clean}
    return info
