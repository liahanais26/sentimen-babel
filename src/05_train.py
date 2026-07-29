"""
05_train.py

Tahapan:
1. Membaca dataset hasil labeling
2. Membagi data train dan test
3. TF-IDF Vectorization
4. Training model SVM
5. Menyimpan model dan TF-IDF Vectorizer
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ======================================================
# Menambahkan root project ke Python Path
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LABELED_DATA,
    SVM_MODEL,
    TFIDF_MODEL,
    RANDOM_STATE,
    TEST_SIZE,
)


def timer(start):
    return f"{time.time()-start:.2f} detik"


# ======================================================
# Main
# ======================================================

def main():

    total_start = time.time()

    print("=" * 60)
    print("05. TRAINING MODEL SVM")
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

    print(f"Jumlah data awal : {len(df)}")

    # ==================================================
    # Filter hanya Positif dan Negatif
    # ==================================================

    df = df[df["label"].isin(["Positif", "Negatif"])].copy()

    print(f"Jumlah data setelah filter : {len(df)}")

    print("\nDistribusi Label")
    print(df["label"].value_counts())

    # ==================================================
    # Missing Value
    # ==================================================

    missing = df["text_final"].isna().sum()

    print(f"Missing text_final : {missing}")

    if missing > 0:

        df = df.dropna(subset=["text_final"])

        print(f"Sisa data : {len(df)}")

    print("\nDistribusi Label Setelah Cleaning")
    print(df["label"].value_counts())

    # ==================================================
    # Train Test Split
    # ==================================================

    print("\nMembagi data train dan test...")

    X_train, X_test, y_train, y_test = train_test_split(

        df["text_final"],

        df["label"],

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=df["label"]

    )

    print(f"Data train : {len(X_train)}")
    print(f"Data test  : {len(X_test)}")

    print("\nDistribusi Label Train")
    print(y_train.value_counts())

    print("\nDistribusi Label Test")
    print(y_test.value_counts())

    # ==================================================
    # TF-IDF
    # ==================================================

    print("\nMelakukan TF-IDF...")

    start = time.time()

    vectorizer = TfidfVectorizer(

        lowercase=False,

        max_features=3000,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95

    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Selesai ({timer(start)})")

    print(f"Jumlah fitur : {X_train_tfidf.shape[1]}")

    # ==================================================
    # Training Model
    # ==================================================

    print("\nTraining model SVM...")

    start = time.time()

    model = LinearSVC(

        class_weight="balanced",

        random_state=RANDOM_STATE

    )

    model.fit(

        X_train_tfidf,

        y_train

    )

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Simpan Model
    # ==================================================

    SVM_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    TFIDF_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nMenyimpan model...")

    joblib.dump(

        model,

        SVM_MODEL

    )

    joblib.dump(

        vectorizer,

        TFIDF_MODEL

    )

    # ==================================================
    # Ringkasan
    # ==================================================

    print("\n" + "=" * 60)
    print("HASIL TRAINING")
    print("=" * 60)

    print(f"Jumlah data : {len(df)}")
    print(f"Data train : {len(X_train)}")
    print(f"Data test  : {len(X_test)}")

    print(f"Jumlah fitur : {X_train_tfidf.shape[1]}")

    print(f"\nModel SVM : {SVM_MODEL}")

    print(f"TF-IDF Vectorizer : {TFIDF_MODEL}")

    print(f"\nTotal waktu : {timer(total_start)}")


if __name__ == "__main__":
    main()