"""Benchmark metode enhancement area nomor berdasarkan CER.
Pakai: python scripts/evaluate_cer.py --samples data/samples --truth 571012022000056
"""
import argparse, glob, os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ijazah import config as C
from ijazah.preprocess import load_and_orient, to_gray, global_enhance, crop_roi
from ijazah.enhance import ENHANCERS
from ijazah.ocr import ocr_line, extract_number, cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--truth", default="571012022000056", help="ground truth nomor ijazah")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.samples, "*.jp*g")))
    rows = []
    for f in files:
        gray = to_gray(load_and_orient(f))
        roi = crop_roi(global_enhance(gray), C.ROI_NUMBER)
        for name, fn in ENHANCERS.items():
            t0 = time.time()
            raw = ocr_line(fn(roi))
            pred = extract_number(raw)
            rows.append({"image": os.path.basename(f), "method": name, "pred": pred,
                         "CER": cer(pred, a.truth), "exact": pred == a.truth,
                         "sec": time.time() - t0})
        print("done", os.path.basename(f), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/cer_detail.csv", index=False)
    summ = (df.groupby("method")
              .agg(mean_CER=("CER", "mean"), exact_match=("exact", "sum"),
                   worst_CER=("CER", "max"), mean_sec=("sec", "mean"))
              .sort_values(["mean_CER", "worst_CER"]))
    summ["mean_CER_%"] = (summ["mean_CER"] * 100).round(2)
    summ.to_csv(f"{a.out}/cer_summary.csv")
    piv = df.pivot(index="method", columns="image", values="CER").loc[summ.index]
    piv.to_csv(f"{a.out}/cer_per_image.csv")
    print(summ.round(4).to_string())
    print("\nCER per citra (baris = metode):")
    print((piv * 100).round(1).to_string())


if __name__ == "__main__":
    main()
