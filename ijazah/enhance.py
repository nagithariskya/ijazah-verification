"""Kumpulan metode enhancement untuk area nomor ijazah (dibandingkan lewat CER)."""
import cv2
import numpy as np


def _clahe(g, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


def _up(g, f=3):
    return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def _bgnorm(g, k=51):
    """Background normalization: bagi citra dengan estimasi latar (closing + blur)
    -> menghilangkan pola guilloche, shading, dan color-cast."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return cv2.divide(g, bg, scale=255)


def _unsharp(g, sigma=1.5, amount=1.5):
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1 + amount, blur, -amount, 0)


def _otsu(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _adaptive(g):
    bs = max(15, (g.shape[0] // 4) | 1)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, bs, 15)


def _gamma(g, gamma=0.6):
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], np.uint8)
    return cv2.LUT(g, lut)


def _nlm(g):
    return cv2.fastNlMeansDenoising(g, None, 12, 7, 21)


ENHANCERS = {
    "none_gray":              lambda g: g,
    "up3":                    lambda g: _up(g, 3),
    "clahe":                  lambda g: _clahe(g),
    "gaussian_clahe":         lambda g: _clahe(cv2.GaussianBlur(g, (3, 3), 0)),
    "median":                 lambda g: cv2.medianBlur(g, 3),
    "nlm":                    lambda g: _nlm(g),
    "gamma":                  lambda g: _gamma(g),
    "unsharp":                lambda g: _unsharp(g),
    "bgnorm":                 lambda g: _bgnorm(g),
    "otsu":                   lambda g: _otsu(g),
    "adaptive":               lambda g: _adaptive(g),
    "up3_nlm_clahe_sharp":    lambda g: _unsharp(_clahe(_nlm(_up(g, 3)))),
    "up3_clahe_otsu":         lambda g: _otsu(_clahe(_up(g, 3))),
    "bgnorm_up3":             lambda g: _up(_bgnorm(g), 3),
    "bgnorm_up3_otsu":        lambda g: _otsu(_up(_bgnorm(g), 3)),
    "median_bgnorm_up3_otsu": lambda g: _otsu(_up(_bgnorm(cv2.medianBlur(g, 3)), 3)),
}


def enhance_number_roi(gray_roi: np.ndarray, method: str) -> np.ndarray:
    return ENHANCERS[method](gray_roi)
