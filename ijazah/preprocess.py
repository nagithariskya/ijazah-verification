"""Tahap: load -> koreksi orientasi -> grayscale -> enhancement global -> crop ROI."""
import cv2
import numpy as np
from .config import TARGET_LONG_SIDE


def load_and_orient(path: str, rotate: str = "auto") -> np.ndarray:
    """Baca citra; jika portrait (teks tegak lurus), putar 90 derajat searah jarum jam."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if rotate == "cw" or (rotate == "auto" and h > w):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "ccw":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    s = TARGET_LONG_SIDE / max(h, w)
    if abs(s - 1) > 0.02:
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def global_enhance(gray: np.ndarray) -> np.ndarray:
    """Enhancement global ringan: contrast stretching persentil 1-99."""
    lo, hi = np.percentile(gray, (1, 99))
    if hi - lo < 1:
        return gray
    out = (gray.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_roi(img: np.ndarray, roi) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
