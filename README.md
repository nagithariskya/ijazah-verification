# Verifikasi Ijazah: OCR Nomor Ijazah + Deteksi Tanda Tangan

Prototype *miniproject* pengolahan citra: menerima citra ijazah, lalu menghasilkan

```
Input : ijazah_001.jpg
Output:
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
```

## 1. Pipeline

```
Citra Ijazah
   ↓  (koreksi orientasi + normalisasi skala)
Grayscale
   ↓
Image Enhancement (global: contrast stretching persentil 1–99)
   ├────────────────────────────┬──────────────────────────────┐
   ↓                            ↓
Area Nomor Ijazah           Area Tanda Tangan (Rektor & Dekan)
   ↓                            ↓
Enhancement (median filter) Thresholding (normalisasi latar + Otsu terbatas)
   ↓                            ↓
OCR (Tesseract, psm 7)      Morphology (opening 3×3 → closing 9×9)
   ↓                            ↓
Nomor Ijazah                Signature Detection (connected components)
   └────────────────────────────┴──────────────────────────────┘
                         ↓
                  Hasil Verifikasi
```

## 2. Cara menjalankan

### Prasyarat
- Python ≥ 3.10
- **Tesseract OCR** terpasang di sistem
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: pasang dari https://github.com/UB-Mannheim/tesseract/wiki (lokasi default).
    Program mencari `tesseract.exe` otomatis (PATH, `Program Files`, `AppData`). Jika lokasinya tidak standar:
    `$env:TESSERACT_CMD = "D:\\lokasi\\tesseract.exe"` (PowerShell) sebelum menjalankan program.

### Instalasi
```bash
git clone <https://github.com/nagithariskya/ijazah-verification.git>
cd ijazah-verification
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Menjalankan verifikasi
```bash
python verify.py data/samples/01_HighQuality_Enhanced.jpg
```
Contoh keluaran:
```
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
  - rektor : PRESENT (ink=0.0331, width=0.772)
  - dekan  : PRESENT (ink=0.0975, width=0.625)
```
Opsi: `--enhancer NAMA` (lihat `ijazah/enhance.py`), `--rotate auto|cw|ccw|none`,
`--debug-dir results/debug` (simpan citra tiap tahap), `--json`.

### Eksperimen & pengujian (reproduksi tabel di bawah)
Letakkan citra uji di `data/samples/` (tidak ikut di-commit karena berisi data pribadi), lalu:
```bash
python scripts/evaluate_cer.py        # CER tiap metode pada citra uji      -> results/cer_*.csv
python scripts/stress_test_cer.py     # CER pada degradasi sintetis berat   -> results/stress_*.csv
python scripts/test_signature.py      # uji kontrol positif/negatif tanda tangan
```
Ground truth nomor ijazah: `571012022000056` (argumen `--truth`).

## 3. Struktur repositori
```
verify.py                 CLI utama
ijazah/config.py          ROI relatif, parameter, metode enhancement default
ijazah/preprocess.py      orientasi, grayscale, enhancement global, crop ROI
ijazah/enhance.py         16 metode enhancement area nomor
ijazah/ocr.py             Tesseract, ekstraksi digit, CER (Levenshtein)
ijazah/signature.py       thresholding → morphology → deteksi tanda tangan
scripts/                  evaluasi CER, stress test, uji tanda tangan
results/                  CSV hasil eksperimen + citra debug
```

## 4. Penjelasan metode

| Tahap | Metode | Alasan |
|---|---|---|
| Orientasi | Citra portrait → rotasi 90° CW; skala sisi terpanjang = 2400 px | ROI relatif konsisten antar resolusi (citra uji 1132–3506 px) |
| Grayscale | `cv2.cvtColor` BGR→GRAY | Membuang pengaruh *color shift*/tint; OCR & thresholding tidak butuh warna |
| Enhancement global | Contrast stretching persentil 1–99 | Memulihkan citra *low-contrast / faded / underexposed* secara murah |
| ROI | Koordinat relatif (nomor, Rektor, Dekan) | Layout ijazah tetap; cepat dan deterministik |
| Enhancement nomor | **Median filter 3×3** (dipilih dari CER, lihat §5) | Menekan *salt-and-pepper*/noise tanpa merusak tepi angka |
| OCR | Tesseract 5 (`--oem 3 --psm 7`), ambil deret digit terpanjang ≥ 10 digit | Satu baris teks; *post-processing* regex menghilangkan label "Nomor ijazah:" |
| Thresholding TTD | Normalisasi latar (closing + blur lalu pembagian) → Otsu dibatasi (≤200 dan ≥40 di bawah median) | Menghilangkan pola guilloche & warna kertas; batas mencegah ROI kosong "menyala" |
| Morphology TTD | Opening 3×3 (buang bintik) → Closing 9×9 (sambung goresan putus) | Goresan tanda tangan menjadi komponen utuh |
| Signature detection | Connected components: rasio tinta ≥ 0,4 % **dan** lebar komponen terbesar ≥ 35 % lebar ROI | Tanda tangan = goresan lebar & menyambung; noise/teks kecil tidak lolos |
| Keputusan | `PRESENT` bila tanda tangan **Rektor dan Dekan** keduanya terdeteksi | Ijazah ini ditandatangani keduanya (ijazah sekolah: ubah ROI ke Kepala Sekolah di `config.py`) |

CER dihitung pada nomor hasil ekstraksi: `CER = (S + D + I) / N`, N = 15 digit ground truth (jarak Levenshtein).

## 5. Metode enhancement mana yang paling efektif (berdasarkan CER)?

### 5a. Pada 9 citra uji asli (`results/cer_summary.csv`)
**15 dari 16 metode menghasilkan CER 0 % (9/9 nomor benar).** Hanya `adaptive` threshold yang buruk
(CER rata-rata 11,85 %; gagal total pada `04_HighNoise`: 100 %, dan 6,7 % pada `03_Blurred`).
Artinya: pada degradasi seperti di dataset ini, enhancement global + Tesseract sudah cukup, sehingga
**data asli tidak membedakan** metode-metode lainnya. Menyimpulkan "paling efektif" hanya dari tabel ini tidak sahih.

### 5b. Stress test degradasi sintetis berat (`results/stress_summary.csv`)
Satu citra bersih didegradasi (blur σ=3/5, noise σ=30/60 [3 seed], low-contrast, faded, low-res 0,25/0,18,
JPEG q=5, shading tak merata, kombinasi) → 18 kasus per metode.

| Peringkat | Metode | Mean CER | Exact match (dari 18) |
|---|---|---|---|
| 1 | **median** | **13,3 %** | 12 |
| 2 | median + bgnorm + up3 + Otsu | 15,2 % | 11 |
| 3 | gamma | 23,0 % | 12 |
| 4 | gaussian + CLAHE | 23,3 % | 11 |
| 5 | none (gray + stretching) | 30,0 % | 9 |
| … | otsu / nlm / clahe / up3 / unsharp | 30,7 – 42,2 % | 6–10 |
| 16 | adaptive threshold | 57,0 % | 6 |

**Kesimpulan:** **Median filter** memiliki CER terendah. Keunggulannya datang dari **ketahanan terhadap noise**
(pada noise σ=30: CER 0 % vs 33–100 % metode lain). Metode lain yang tampak canggih (CLAHE, upscaling ×3,
unsharp, binarisasi) justru *memperkuat* noise sehingga CER memburuk. Pada blur, low-contrast, faded, JPEG,
shading, semua metode setara (CER ≈ 0 %) karena contrast stretching global sudah menanganinya.

### 5c. Keterbatasan (jujur)
- Stress test memakai **satu** citra dasar, noise hanya 3 seed → selisih kecil antar peringkat bawah tidak signifikan secara statistik; yang robust: median ≫ adaptive/CLAHE/up3 pada noise.
- Blur σ=5 membuat **semua** metode gagal (CER 100 %) → batas kemampuan Tesseract tanpa *deblurring*/super-resolution.
- Default awal saya (`bgnorm_up3_otsu`) dipilih sebelum eksperimen; stress test menunjukkan ia kalah dari `median` (40,7 % vs 13,3 %), sehingga default diganti ke `median`.
- ROI bersifat tetap (relatif): ijazah dengan layout/crop berbeda perlu penyesuaian `ijazah/config.py`, atau tambahkan lokalisasi otomatis (mis. cari kata kunci "Nomor" lewat `image_to_data`).
- Deteksi tanda tangan memeriksa **keberadaan** goresan, bukan **keaslian**. Coretan apa pun di area itu dianggap PRESENT.

## 6. Hasil pada data uji
- Nomor ijazah: 9/9 citra terbaca `571012022000056` (CER 0 %), lihat `results/predictions.csv`.
- Tanda tangan: kontrol positif 9/9 `PRESENT`; kontrol negatif sintetis (area TTD diganti kertas kosong: tanpa Rektor / tanpa Dekan / tanpa keduanya) 27/27 `NOT PRESENT` → total 36/36 (`scripts/test_signature.py`). Kontrol negatif bersifat sintetis, bukan ijazah tanpa TTD sungguhan.

## 7. Catatan privasi
Citra ijazah memuat data pribadi (nama, tanggal lahir, NIM, foto). Folder `data/samples/` di-*gitignore*;
jangan mengunggahnya ke repositori publik.
