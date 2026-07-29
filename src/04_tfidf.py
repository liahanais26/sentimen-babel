"""
04_tfidf.py

Tahapan:
1. Membaca dataset hasil labeling
2. Melakukan TF-IDF Vectorization
3. Menyimpan hasil TF-IDF
4. Menyimpan model TF-IDF Vectorizer
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================================================
# Menambahkan root project ke Python Path
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LABELED_DATA,
    TFIDF_DATA,
)


def timer(start):
    return f"{time.time()-start:.2f} detik"


# ======================================================
# Main
# ======================================================

def main():

    total_start = time.time()

    print("=" * 60)
    print("04. TF-IDF VECTORIZATION")
    print("=" * 60)

    # ==================================================
    # Cek File
    # ==================================================

    if not LABELED_DATA.exists():
        raise FileNotFoundError(
            f"\nDataset tidak ditemukan:\n{LABELED_DATA}"
        )

    # ==================================================
    # Membaca Dataset
    # ==================================================

    print("\nMembaca dataset...")

    df = pd.read_excel(LABELED_DATA)

    print(f"Jumlah data : {len(df)}")

    # ==================================================
    # Missing Value
    # ==================================================

    print("\nCek Missing Value")

    missing = df["text_final"].isna().sum()

    print(f"Missing text_final : {missing}")

    if missing > 0:

        print("Menghapus data yang kosong...")

        df = df.dropna(subset=["text_final"])

        print(f"Sisa data : {len(df)}")

    # ==================================================
    # TF-IDF
    # ==================================================

    print("\nMelakukan TF-IDF Vectorization...")

    start = time.time()

    vectorizer = TfidfVectorizer(

        lowercase=False,

        max_features=3000,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95

    )

    X = vectorizer.fit_transform(df["text_final"])

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Informasi TF-IDF
    # ==================================================

    print("\nInformasi TF-IDF")

    print(f"Jumlah dokumen : {X.shape[0]}")
    print(f"Jumlah fitur   : {X.shape[1]}")

    print("\n20 fitur pertama:")

    print(vectorizer.get_feature_names_out()[:20])

    # ==================================================
    # Konversi ke DataFrame
    # ==================================================

    print("\nMengubah Sparse Matrix menjadi DataFrame...")

    start = time.time()

    tfidf_df = pd.DataFrame(

        X.toarray(),

        columns=vectorizer.get_feature_names_out()

    )

    tfidf_df["label"] = df["label"].values

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Simpan Dataset TF-IDF
    # ==================================================

    TFIDF_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nMenyimpan dataset TF-IDF...")

    tfidf_df.to_csv(
        TFIDF_DATA,
        index=False
    )

    # ==================================================
    # Simpan Vectorizer
    # ==================================================

    model_dir = ROOT_DIR / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    # ==================================================
    # Ringkasan
    # ==================================================

    print("\n" + "=" * 60)
    print("HASIL TF-IDF")
    print("=" * 60)

    print(f"Jumlah dokumen : {X.shape[0]}")
    print(f"Jumlah fitur   : {X.shape[1]}")

    print(f"\nDataset TF-IDF : {TFIDF_DATA}")

    print(f"Vectorizer     : {vectorizer_path}")

    print(f"\nTotal waktu : {timer(total_start)}")


if __name__ == "__main__":
    main()