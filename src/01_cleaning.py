"""
01_cleaning.py

Tahap pertama pipeline:
1. Membaca dataset
2. Menghapus missing value
3. Menghapus duplikasi
4. Menghapus metadata artikel
5. Menyimpan dataset bersih
"""

import re
import sys
from pathlib import Path

import pandas as pd

# ======================================================
# Menambahkan root project ke Python Path
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RAW_DATA, CLEAN_DATA


# ======================================================
# Membersihkan boilerplate berita
# ======================================================

PATTERNS = [

    # Copyright & Disclaimer
    r"COPYRIGHT.*",
    r"Hak Cipta.*",
    r"Dilarang keras.*",

    # Metadata
    r"Pewarta\s*:.*",
    r"Reporter\s*:.*",
    r"Editor\s*:.*",
    r"Uploader\s*:.*",
    r"Redaktur\s*:.*",
    r"Kontributor\s*:.*",
    r"Fotografer\s*:.*",
    r"Foto\s*:.*",
    r"Dok\s*:.*",

    # Link
    r"Baca juga.*",
    r"Selengkapnya.*",

    # URL
    r"http\S+",
    r"www\S+",

    # Email
    r"\S+@\S+",
]


def clean_article(text):

    if pd.isna(text):
        return ""

    text = str(text)

    for pattern in PATTERNS:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    # Hilangkan spasi berlebih

    text = re.sub(r"\n+", "\n", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ======================================================
# Main
# ======================================================

def main():

    print("=" * 60)
    print("01. DATA CLEANING")
    print("=" * 60)

    print(f"Dataset : {RAW_DATA}")

    if not RAW_DATA.exists():

        raise FileNotFoundError(
            f"\nDataset tidak ditemukan:\n{RAW_DATA}"
        )

    df = pd.read_excel(RAW_DATA)

    print(f"\nJumlah data awal : {len(df)}")

    # ============================================
    # Missing Value
    # ============================================

    before = len(df)

    df.dropna(
        subset=["Judul", "Isi"],
        inplace=True
    )

    print(f"Missing Value dihapus : {before-len(df)}")

    # ============================================
    # Duplicate
    # ============================================

    before = len(df)

    df.drop_duplicates(
        subset=["Judul", "Isi"],
        inplace=True
    )

    print(f"Duplikasi dihapus : {before-len(df)}")

    # ============================================
    # Cleaning Isi
    # ============================================

    print("\nMembersihkan metadata artikel...")

    df["Isi"] = df["Isi"].apply(clean_article)

    # ============================================
    # Hapus artikel kosong
    # ============================================

    before = len(df)

    df = df[
        df["Isi"].str.strip() != ""
    ]

    print(f"Artikel kosong dihapus : {before-len(df)}")

    df.reset_index(
        drop=True,
        inplace=True
    )

    print(f"\nJumlah data akhir : {len(df)}")

    # ============================================
    # Simpan
    # ============================================

    CLEAN_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_excel(
        CLEAN_DATA,
        index=False
    )

    print("\nDataset berhasil disimpan.")

    print(CLEAN_DATA)


if __name__ == "__main__":
    main()