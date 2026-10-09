"""OCR nomor ijazah (Tesseract) + ekstraksi digit + metrik CER."""
import os
import re
import shutil
import cv2
import numpy as np
import pytesseract
from .config import MIN_DIGITS


def _find_tesseract():
    """Cari tesseract: variabel lingkungan TESSERACT_CMD -> PATH -> lokasi umum Windows."""
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
              os.path.join(local, "Programs", "Tesseract-OCR", "tesseract.exe")):
        if os.path.isfile(c):
            return c
    return None


_TESS = _find_tesseract()
if _TESS:
    pytesseract.pytesseract.tesseract_cmd = _TESS
else:
    raise SystemExit(
        "Tesseract OCR tidak ditemukan.\n"
        "Pasang dari https://github.com/UB-Mannheim/tesseract/wiki (Windows) "
        "atau 'sudo apt install tesseract-ocr' (Linux), lalu jalankan ulang.\n"
        "Jika lokasinya tidak standar, set variabel lingkungan TESSERACT_CMD "
        "ke path tesseract.exe.")


def _pad(img, p=20):
    v = int(np.median(img))
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=v)


def ocr_line(img: np.ndarray) -> str:
    return pytesseract.image_to_string(_pad(img), lang="eng", config="--oem 3 --psm 7").strip()


def extract_number(text: str) -> str:
    """Ambil deret digit terpanjang (>= MIN_DIGITS) dari hasil OCR."""
    t = text.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    t = re.sub(r"(?<=\d)[ .\-](?=\d)", "", t)
    runs = [r for r in re.findall(r"\d+", t) if len(r) >= MIN_DIGITS]
    return max(runs, key=len) if runs else ""


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    """Character Error Rate = (S + D + I) / N."""
    return levenshtein(pred, truth) / max(1, len(truth))
