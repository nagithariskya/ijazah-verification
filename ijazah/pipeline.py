"""Pipeline: Citra -> Grayscale -> Enhancement -> (ROI nomor | ROI tanda tangan) -> Hasil."""
import os
import cv2
from . import config as C
from .preprocess import load_and_orient, to_gray, global_enhance, crop_roi
from .enhance import enhance_number_roi
from .ocr import ocr_line, extract_number
from .signature import detect_signature


def verify_ijazah(path, enhancer=C.DEFAULT_NUMBER_ENHANCER, rotate="auto", debug_dir=None):
    color = load_and_orient(path, rotate)
    gray = to_gray(color)
    enh = global_enhance(gray)

    # cabang 1: nomor ijazah
    num_roi = crop_roi(enh, C.ROI_NUMBER)
    num_enh = enhance_number_roi(num_roi, enhancer)
    raw = ocr_line(num_enh)
    number = extract_number(raw)

    # cabang 2: tanda tangan (Rektor & Dekan)
    sig, dbg = {}, {}
    for name, roi in (("rektor", C.ROI_REKTOR), ("dekan", C.ROI_DEKAN)):
        info, d = detect_signature(crop_roi(enh, roi), return_debug=True)
        sig[name], dbg[name] = info, d

    present = all(s["present"] for s in sig.values())
    result = {
        "file": os.path.basename(path),
        "nomor_ijazah": number or "TIDAK TERBACA",
        "ocr_raw": raw,
        "tanda_tangan": "PRESENT" if present else "NOT PRESENT",
        "detail_tanda_tangan": sig,
        "enhancer": enhancer,
    }
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(f"{debug_dir}/{stem}_number_roi.png", num_roi)
        cv2.imwrite(f"{debug_dir}/{stem}_number_enh.png", num_enh)
        for name, d in dbg.items():
            cv2.imwrite(f"{debug_dir}/{stem}_{name}_binary.png", d["binary"])
            cv2.imwrite(f"{debug_dir}/{stem}_{name}_morph.png", d["morphology"])
    return result
