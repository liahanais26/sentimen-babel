"""
02_preprocessing.py

Tahapan preprocessing:
1. Case Folding
2. Cleaning
3. Tokenizing
4. Stopword Removal
5. Stemming
6. Join Token
"""

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ======================================================
# Menambahkan root project ke Python Path
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CLEAN_DATA, PREPROCESS_DATA

from src.utils import (
    clean_text,
    tokenize,
    remove_stopwords,
    stemming,
    join_tokens
)

tqdm.pandas()


def timer(start):
    return f"{time.time()-start:.2f} detik"


def main():

    total_start = time.time()

    print("=" * 60)
    print("02. PREPROCESSING DATASET")
    print("=" * 60)

    # ==================================================
    # Cek Dataset
    # ==================================================

    if not CLEAN_DATA.exists():
        raise FileNotFoundError(
            f"\nDataset tidak ditemukan:\n{CLEAN_DATA}"
        )

    print("\nMembaca dataset...")

    df = pd.read_excel(CLEAN_DATA)

    print(f"Jumlah data : {len(df)}")

    # ==================================================
    # Gabungkan Judul + Isi
    # ==================================================

    df["text"] = (
        df["Judul"].fillna("") + " " +
        df["Isi"].fillna("")
    )

    # ==================================================
    # Case Folding + Cleaning
    # ==================================================

    print("\n[1/5] Case Folding & Cleaning")

    start = time.time()

    df["clean"] = df["text"].progress_apply(clean_text)

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Tokenizing
    # ==================================================

    print("\n[2/5] Tokenizing")

    start = time.time()

    df["token"] = df["clean"].progress_apply(tokenize)

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Stopword Removal
    # ==================================================

    print("\n[3/5] Stopword Removal")

    start = time.time()

    df["stopword"] = df["token"].progress_apply(remove_stopwords)

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Stemming
    # ==================================================

    print("\n[4/5] Stemming")

    start = time.time()

    df["stemming"] = df["stopword"].progress_apply(stemming)

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Join Token
    # ==================================================

    print("\n[5/5] Join Token")

    start = time.time()

    df["text_final"] = df["stemming"].progress_apply(join_tokens)

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Hapus hasil preprocessing kosong
    # ==================================================

    before = len(df)

    df = df[
        df["text_final"].str.strip() != ""
    ].copy()

    print(f"\nData kosong setelah preprocessing : {before-len(df)}")

    df.reset_index(drop=True, inplace=True)

    # ==================================================
    # Simpan
    # ==================================================

    PREPROCESS_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nMenyimpan dataset...")

    df.to_excel(
        PREPROCESS_DATA,
        index=False
    )

    print("Dataset berhasil disimpan.")

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI")
    print("=" * 60)

    print(f"Output      : {PREPROCESS_DATA}")
    print(f"Jumlah data : {len(df)}")
    print(f"Total waktu : {timer(total_start)}")


if __name__ == "__main__":
    main()