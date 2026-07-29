"""
03_labeling.py

Tahapan:
1. Membaca dataset preprocessing
2. Membaca lexicon positif & negatif
3. Menghitung skor sentimen
4. Memberikan label
5. Menyimpan dataset
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

from config import (
    PREPROCESS_DATA,
    LABELED_DATA,
    POSITIVE_LEXICON,
    NEGATIVE_LEXICON,
)

tqdm.pandas()


def timer(start):
    return f"{time.time()-start:.2f} detik"


# ======================================================
# Fungsi Menghitung Skor Sentimen
# ======================================================

def sentiment_score(text):

    score = 0

    if pd.isna(text):
        return score

    words = text.split()

    for word in words:

        # Bobot positif pada INSET bernilai positif
        score += positive_dict.get(word, 0)

        # Bobot negatif pada INSET SUDAH bernilai negatif
        score += negative_dict.get(word, 0)

    return score


# ======================================================
# Fungsi Label
# ======================================================

def sentiment_label(score):

    if score > 0:
        return "Positif"

    elif score < 0:
        return "Negatif"

    else:
        return "Netral"

# ======================================================
# Fungsi Menampilkan Kontributor Skor
# ======================================================

def explain_score(text):

    hasil = []

    if pd.isna(text):
        return hasil

    for word in text.split():

        if word in positive_dict:
            hasil.append(
                (word, positive_dict[word], "Positif")
            )

        elif word in negative_dict:
            hasil.append(
                (word, negative_dict[word], "Negatif")
            )

    return hasil


# ======================================================
# Main
# ======================================================

def main():

    total_start = time.time()

    print("=" * 60)
    print("03. LABELING SENTIMEN")
    print("=" * 60)

    # ==================================================
    # Cek File
    # ==================================================

    if not PREPROCESS_DATA.exists():
        raise FileNotFoundError(
            f"\nDataset tidak ditemukan:\n{PREPROCESS_DATA}"
        )

    if not POSITIVE_LEXICON.exists():
        raise FileNotFoundError(
            f"\nLexicon positif tidak ditemukan:\n{POSITIVE_LEXICON}"
        )

    if not NEGATIVE_LEXICON.exists():
        raise FileNotFoundError(
            f"\nLexicon negatif tidak ditemukan:\n{NEGATIVE_LEXICON}"
        )

    # ==================================================
    # Membaca Dataset
    # ==================================================

    print("\nMembaca dataset preprocessing...")

    df = pd.read_excel(PREPROCESS_DATA)

    print(f"Jumlah data : {len(df)}")

    # ==================================================
    # Membaca Lexicon
    # ==================================================

    print("\nMembaca lexicon...")

    positive = pd.read_csv(
        POSITIVE_LEXICON,
        sep="\t"
    )

    negative = pd.read_csv(
        NEGATIVE_LEXICON,
        sep="\t"
    )

    print(f"Kata positif : {len(positive)}")
    print(f"Kata negatif : {len(negative)}")

    # ==================================================
    # Dictionary
    # ==================================================

    global positive_dict
    global negative_dict

    positive_dict = dict(
        zip(
            positive["word"],
            positive["weight"]
        )
    )

    negative_dict = dict(
        zip(
            negative["word"],
            negative["weight"]
        )
    )

    print("\nLexicon berhasil dimuat.")

    # ==================================================
    # Hitung Skor
    # ==================================================

    print("\nMenghitung skor sentimen...")

    start = time.time()

    df["score"] = df["text_final"].progress_apply(
        sentiment_score
    )

    print(f"Selesai ({timer(start)})")

    # ==================================================
    # Debug Distribusi Skor
    # ==================================================

    print("\nDistribusi skor sentimen")

    print(df["score"].describe())

    print("\nFrekuensi skor")

    print(df["score"].value_counts().sort_index())

    # ==================================================
    # Label
    # ==================================================

    print("\nMemberikan label...")

    df["label"] = df["score"].apply(
        sentiment_label
    )

    # ==================================================
    # Simpan
    # ==================================================

    LABELED_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nMenyimpan dataset...")

    df.to_excel(
        LABELED_DATA,
        index=False
    )

    # ==================================================
    # Ringkasan
    # ==================================================

    print("\n" + "=" * 60)
    print("HASIL LABELING")
    print("=" * 60)

    print(df["label"].value_counts())

    print("\nContoh hasil labeling:")

    print(
        df[
            [
                "text_final",
                "score",
                "label"
            ]
        ].head(10)
    )

    # ==================================================
    # Analisis Kontributor Skor
    # ==================================================

    print("\n" + "=" * 60)
    print("ANALISIS KONTRIBUTOR SKOR")
    print("=" * 60)

    # Menampilkan analisis untuk 5 data pertama
    for index in range(min(5, len(df))):

        print(f"\nData ke-{index+1}")
        print("-" * 60)

        print("Label :", df.loc[index, "label"])
        print("Score :", df.loc[index, "score"])

        print("\nText Final:")
        print(df.loc[index, "text_final"])

        kontribusi = explain_score(df.loc[index, "text_final"])

        print(f"Jumlah kata yang cocok dengan lexicon : {len(kontribusi)}")

        print("\nKontributor skor:")

        if len(kontribusi) == 0:
            print("Tidak ada kata yang cocok dengan lexicon.")

        else:
            for kata, bobot, tipe in kontribusi:
                print(f"{kata:<20} {bobot:>3} ({tipe})")

    print(f"\nOutput : {LABELED_DATA}")

    print(f"Total waktu : {timer(total_start)}")


if __name__ == "__main__":
    main()