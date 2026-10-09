"""CLI: python verify.py path/ke/ijazah.jpg [--enhancer NAMA] [--debug-dir DIR]"""
import argparse
import json
from ijazah.pipeline import verify_ijazah
from ijazah.enhance import ENHANCERS
from ijazah import config as C


def main():
    ap = argparse.ArgumentParser(description="Verifikasi ijazah: OCR nomor + deteksi tanda tangan")
    ap.add_argument("image")
    ap.add_argument("--enhancer", default=C.DEFAULT_NUMBER_ENHANCER, choices=ENHANCERS.keys())
    ap.add_argument("--rotate", default="auto", choices=["auto", "cw", "ccw", "none"])
    ap.add_argument("--debug-dir", default=None, help="simpan citra tiap tahap")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = verify_ijazah(a.image, a.enhancer, a.rotate, a.debug_dir)
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print(f"Nomor Ijazah : {r['nomor_ijazah']}")
        print(f"Tanda Tangan : {r['tanda_tangan']}")
        for k, v in r["detail_tanda_tangan"].items():
            print(f"  - {k:7s}: {'PRESENT' if v['present'] else 'NOT PRESENT'} "
                  f"(ink={v['ink_ratio']}, width={v['max_width_ratio']})")


if __name__ == "__main__":
    main()
