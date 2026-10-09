"""Stress test: degradasi sintetis BERAT pada citra bersih agar CER antar metode enhancement
terbedakan (pada 9 sampel asli hampir semua metode sudah CER 0%).
Pakai: python scripts/stress_test_cer.py --base data/samples/01_HighQuality_Enhanced.jpg
"""
import argparse, os, sys
import cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ijazah import config as C
from ijazah.preprocess import load_and_orient, to_gray, global_enhance, crop_roi
from ijazah.enhance import ENHANCERS
from ijazah.ocr import ocr_line, extract_number, cer


def blur(s):      return lambda g, r: cv2.GaussianBlur(g, (0, 0), s)
def noise(s):     return lambda g, r: np.clip(g + r.normal(0, s, g.shape), 0, 255).astype(np.uint8)
def lowcon(a):    return lambda g, r: np.clip(128 + a * (g.astype(np.float32) - 128), 0, 255).astype(np.uint8)
def faded(a, b):  return lambda g, r: np.clip(a * g.astype(np.float32) + b, 0, 255).astype(np.uint8)
def lowres(f):
    def fn(g, r):
        h, w = g.shape
        s = cv2.resize(g, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    return fn
def jpeg(q):
    def fn(g, r):
        _, b = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(b, cv2.IMREAD_GRAYSCALE)
    return fn
def chain(*fs):
    def fn(g, r):
        for f in fs: g = f(g, r)
        return g
    return fn
def shade(g, r):  # pencahayaan tidak merata (gradien)
    h, w = g.shape
    grad = np.tile(np.linspace(0.45, 1.0, w, dtype=np.float32), (h, 1))
    return np.clip(g * grad, 0, 255).astype(np.uint8)

DEGRADATIONS = {
    "blur_s3": blur(3), "blur_s5": blur(5),
    "noise_s30": noise(30), "noise_s60": noise(60),
    "lowcontrast_0.25": lowcon(0.25), "lowcontrast_0.15": lowcon(0.15),
    "faded_0.4+120": faded(0.4, 120),
    "lowres_0.25": lowres(0.25), "lowres_0.18": lowres(0.18),
    "jpeg_q5": jpeg(5),
    "uneven_shading": shade,
    "combined": chain(lowres(0.35), blur(1.5), noise(25), lowcon(0.4), jpeg(15)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/samples/01_HighQuality_Enhanced.jpg")
    ap.add_argument("--truth", default="571012022000056")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    base = to_gray(load_and_orient(a.base))
    rows = []
    for dname, dfn in DEGRADATIONS.items():
        for seed in range(a.seeds if "noise" in dname or dname == "combined" else 1):
            g = dfn(base, np.random.default_rng(seed))
            roi = crop_roi(global_enhance(g), C.ROI_NUMBER)
            for m, fn in ENHANCERS.items():
                pred = extract_number(ocr_line(fn(roi)))
                rows.append({"degradation": dname, "seed": seed, "method": m,
                             "pred": pred, "CER": cer(pred, a.truth)})
        print("done", dname, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/stress_detail.csv", index=False)
    s = (df.groupby("method").agg(mean_CER=("CER", "mean"), median_CER=("CER", "median"),
                                  worst_CER=("CER", "max"),
                                  exact=("CER", lambda x: int((x == 0).sum())), n=("CER", "size"))
           .sort_values(["mean_CER", "worst_CER"]))
    s["mean_CER_%"] = (s["mean_CER"] * 100).round(2)
    s.to_csv(f"{a.out}/stress_summary.csv")
    piv = df.groupby(["method", "degradation"])["CER"].mean().unstack().loc[s.index]
    piv.to_csv(f"{a.out}/stress_per_degradation.csv")
    print(s.round(4).to_string()); print()
    print((piv * 100).round(1).T.to_string())


if __name__ == "__main__":
    main()
