"""Uji deteksi tanda tangan: citra asli (harus PRESENT) vs kontrol negatif sintetis
(area tanda tangan diganti potongan kertas kosong -> harus NOT PRESENT).
Pakai: python scripts/test_signature.py --samples data/samples
"""
import argparse, glob, os, sys, tempfile
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ijazah import config as C
from ijazah.pipeline import verify_ijazah
from ijazah.preprocess import load_and_orient

BLANK = (0.02, 0.06, 0.34, 0.26)   # area kertas kosong (kiri-atas)


def roi_px(img, roi):
    h, w = img.shape[:2]
    return int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)


def blank_out(img, rois):
    out = img.copy()
    bx0, by0, _, _ = roi_px(img, BLANK)
    for r in rois:
        x0, y0, x1, y1 = roi_px(img, r)
        out[y0:y1, x0:x1] = img[by0:by0 + (y1 - y0), bx0:bx0 + (x1 - x0)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--samples", default="data/samples")
    a = ap.parse_args()
    cases = {"asli": [], "tanpa_rektor": [C.ROI_REKTOR], "tanpa_dekan": [C.ROI_DEKAN],
             "tanpa_keduanya": [C.ROI_REKTOR, C.ROI_DEKAN]}
    expect = {"asli": "PRESENT", "tanpa_rektor": "NOT PRESENT",
              "tanpa_dekan": "NOT PRESENT", "tanpa_keduanya": "NOT PRESENT"}
    ok = tot = 0
    for f in sorted(glob.glob(os.path.join(a.samples, "*.jp*g"))):
        img = load_and_orient(f)
        line = []
        for name, rois in cases.items():
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "t.png"); cv2.imwrite(p, blank_out(img, rois))
                got = verify_ijazah(p)["tanda_tangan"]
            ok += got == expect[name]; tot += 1
            line.append(f"{name}={got}{'' if got == expect[name] else ' (SALAH)'}")
        print(os.path.basename(f), "|", " | ".join(line))
    print(f"\nAkurasi deteksi tanda tangan: {ok}/{tot} = {ok / tot:.1%}")


if __name__ == "__main__":
    main()
