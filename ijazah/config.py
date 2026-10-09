"""Konfigurasi ROI (koordinat relatif terhadap halaman LANDSCAPE) dan parameter."""

# Sisi terpanjang citra dinormalisasi ke ukuran ini agar skala ROI/OCR konsisten
TARGET_LONG_SIDE = 2400

# ROI = (x0, y0, x1, y1) dalam fraksi lebar/tinggi halaman setelah orientasi dikoreksi
ROI_NUMBER = (0.04, 0.88, 0.34, 0.95)      # baris "Nomor ijazah: ..."
ROI_REKTOR = (0.12, 0.70, 0.42, 0.83)      # tanda tangan Rektor
ROI_DEKAN = (0.63, 0.705, 0.92, 0.825)     # tanda tangan Dekan

# Metode enhancement area nomor yang dipakai sistem (hasil benchmark CER)
DEFAULT_NUMBER_ENHANCER = "median"

# Parameter deteksi tanda tangan
SIG_MIN_INK_RATIO = 0.004     # minimal rasio piksel tinta pada ROI
SIG_MIN_WIDTH_RATIO = 0.35    # lebar komponen terbesar >= 35% lebar ROI
SIG_MIN_COMPONENT_AREA = 150  # piksel

# Format nomor ijazah minimal (jumlah digit)
MIN_DIGITS = 10
