# ==========================================
# VISUALISASI DISTRIBUSI SENTIMEN
# ==========================================

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# Menambahkan root project ke Python Path
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LABELED_DATA, RESULT_DIR

from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------------------
# Membaca Dataset
# ------------------------------------------

print("Membaca dataset...")

df = pd.read_excel(LABELED_DATA)

print(f"Jumlah data awal : {len(df)}")

# ==========================================
# Filter hanya Positif dan Negatif
# ==========================================

df = df[df["label"].isin(["Positif", "Negatif"])].copy()

print(f"Jumlah data setelah filter : {len(df)}")

print("\nDistribusi Label")

print(df["label"].value_counts())

# ------------------------------------------
# Hitung Jumlah Label
# ------------------------------------------

sentiment = df["label"].value_counts()

# ------------------------------------------
# Membuat Grafik
# ------------------------------------------

plt.figure(figsize=(8,6))

colors = []

for label in sentiment.index:

    if label == "Positif":
        colors.append("forestgreen")

    else:
        colors.append("tomato")

bars = plt.bar(
    sentiment.index,
    sentiment.values,
    color=colors,
    edgecolor="black"
)

plt.title(
    "Distribusi Sentimen Berita Pelayanan Publik Bangka Belitung",
    fontsize=14,
    weight="bold"
)

plt.xlabel("Sentimen")
plt.ylabel("Jumlah Berita")

# menampilkan angka di atas batang

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        height+3,
        f"{int(height)}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    RESULT_DIR/"1_distribusi_sentimen.png",
    dpi=300
)

plt.show()

print("✓ Distribusi sentimen berhasil disimpan.")

# ==========================================
# WORD CLOUD POSITIF
# ==========================================

from wordcloud import WordCloud

print("\nMembuat WordCloud Positif...")

# Ambil berita positif
positive_text = " ".join(
    df[df["label"] == "Positif"]["text_final"].astype(str)
)

wordcloud_positive = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="Greens",
    max_words=100
).generate(positive_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud_positive, interpolation="bilinear")
plt.axis("off")

plt.title(
    "WordCloud Sentimen Positif",
    fontsize=16,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "2_wordcloud_positif.png",
    dpi=300
)

plt.show()

print("✓ WordCloud Positif berhasil disimpan.")

# ==========================================
# WORD CLOUD NEGATIF
# ==========================================

print("\nMembuat WordCloud Negatif...")

negative_text = " ".join(
    df[df["label"] == "Negatif"]["text_final"].astype(str)
)

wordcloud_negative = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="Reds",
    max_words=100
).generate(negative_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud_negative, interpolation="bilinear")
plt.axis("off")

plt.title(
    "WordCloud Sentimen Negatif",
    fontsize=16,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "3_wordcloud_negatif.png",
    dpi=300
)

plt.show()

print("✓ WordCloud Negatif berhasil disimpan.")

def plot_top_words(text_series, title, filename, color):

    vectorizer = CountVectorizer(stop_words=None)

    X = vectorizer.fit_transform(text_series)

    word_counts = X.sum(axis=0).A1

    words = vectorizer.get_feature_names_out()

    freq = (
        pd.DataFrame({
            "word": words,
            "count": word_counts
        })
        .sort_values("count", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10,8))

    plt.barh(
        freq["word"][::-1],
        freq["count"][::-1],
        color=color
    )

    plt.title(title, fontsize=15, weight="bold")

    plt.xlabel("Frekuensi")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / filename,
        dpi=300
    )

    plt.show()

    print(f"✓ {filename} berhasil disimpan.")

plot_top_words(

    df[df["label"]=="Positif"]["text_final"],

    "Top 20 Kata Sentimen Positif",

    "4_top20_kata_positif.png",

    "green"
)

print("\nMembuat Top 20 Kata Negatif...")

plot_top_words(

    df[df["label"]=="Negatif"]["text_final"],

    "Top 20 Kata Sentimen Negatif",

    "5_top20_kata_negatif.png",

    "red"
)

# ==========================================
# FUNGSI TOP BIGRAM
# ==========================================

def plot_top_bigrams(text_series, title, filename, color):

    vectorizer = CountVectorizer(
        ngram_range=(2, 2)
    )

    X = vectorizer.fit_transform(text_series)

    counts = X.sum(axis=0).A1

    bigrams = vectorizer.get_feature_names_out()

    freq = (
        pd.DataFrame({
            "Bigram": bigrams,
            "Frekuensi": counts
        })
        .sort_values("Frekuensi", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(12,8))

    plt.barh(
        freq["Bigram"][::-1],
        freq["Frekuensi"][::-1],
        color=color
    )

    plt.title(title, fontsize=15, weight="bold")

    plt.xlabel("Frekuensi")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / filename,
        dpi=300
    )

    plt.show()

    print(f"✓ {filename} berhasil disimpan.")

    # ==========================================
# TOP 20 BIGRAM POSITIF
# ==========================================

print("\nMembuat Top 20 Bigram Positif...")

plot_top_bigrams(

    df[df["label"]=="Positif"]["text_final"],

    "Top 20 Bigram Sentimen Positif",

    "6_top20_bigram_positif.png",

    "green"

)

# ==========================================
# TOP 20 BIGRAM NEGATIF
# ==========================================

print("\nMembuat Top 20 Bigram Negatif...")

plot_top_bigrams(

    df[df["label"]=="Negatif"]["text_final"],

    "Top 20 Bigram Sentimen Negatif",

    "7_top20_bigram_negatif.png",

    "red"

)

# ==========================================
# FUNGSI ANALISIS TOPIK KELUHAN
# ==========================================

def analyze_topics(df_sentiment):

    topic_dict = {

        "Administrasi Kependudukan": [
            "ktp", "kk", "akta", "dukcapil",
            "kependudukan", "nik", "dokumen",
            "administrasi", "catatan sipil"
        ],

        "Kesehatan": [
            "rumah sakit", "rsud", "puskesmas",
            "dokter", "pasien", "obat",
            "bpjs", "kesehatan", "rawat",
            "ambulans", "vaksin"
        ],

        "Pendidikan": [
            "sekolah", "guru", "siswa",
            "pendidikan", "kampus",
            "beasiswa", "kelas", "murid"
        ],

        "Infrastruktur": [
            "jalan", "jembatan", "drainase",
            "trotoar", "aspal", "lampu jalan",
            "infrastruktur", "rusak"
        ],

        "Transportasi": [
            "terminal", "angkutan",
            "transportasi", "pelabuhan",
            "bandara", "bus", "jalan raya"
        ],

        "Perizinan": [
            "izin", "perizinan", "oss",
            "usaha", "legalitas", "nib",
            "investasi"
        ],

        "Lingkungan": [
            "sampah", "limbah", "banjir",
            "air bersih", "lingkungan",
            "kebersihan"
        ],

        "Sosial": [
            "bansos", "bantuan",
            "pkh", "kemiskinan",
            "disabilitas", "lansia"
        ],

        "Keamanan": [
            "satpol", "polisi",
            "keamanan", "kriminal",
            "ketertiban"
        ],

        "Perpajakan & Retribusi": [
            "pajak", "retribusi",
            "pendapatan daerah",
            "pbb", "pajak daerah"
        ],

        "Ketenagakerjaan": [
            "tenaga kerja", "pekerja",
            "buruh", "upah",
            "pelatihan kerja",
            "pengangguran"
        ],

        "Pertanian & Perikanan": [
            "petani", "pertanian",
            "nelayan", "perkebunan",
            "pupuk", "panen", "ikan"
        ],

        "UMKM & Ekonomi": [
            "umkm", "usaha mikro",
            "pasar", "pedagang",
            "ekonomi", "koperasi"
        ],

        "Pariwisata": [
            "wisata", "pariwisata",
            "objek wisata",
            "hotel",
            "destinasi"
        ],

        "Pelayanan Publik Umum": [
            "pelayanan publik",
            "pelayanan",
            "pengaduan",
            "ombudsman"
        ]

    }

    hasil = []

    for topic, keywords in topic_dict.items():

        jumlah_berita = 0

        for text in df_sentiment["text_final"].astype(str):

            text = text.lower()

            if any(keyword in text for keyword in keywords):

                jumlah_berita += 1

        hasil.append({
            "Topik": topic,
            "Jumlah Berita": jumlah_berita
        })

    topic_df = pd.DataFrame(hasil)

    topic_df = topic_df.sort_values(
        by="Jumlah Berita",
        ascending=False
    )

    return topic_df

# ==========================================
# TOPIK POSITIF
# ==========================================

print("\nMenganalisis Topik Positif...")

positive_df = df[df["label"] == "Positif"]

positive_topic_df = analyze_topics(positive_df)

plt.figure(figsize=(12,7))

plt.bar(
    positive_topic_df["Topik"],
    positive_topic_df["Jumlah Berita"],
    color="forestgreen"
)

plt.title(
    "Topik pada Berita Bersentimen Positif",
    fontsize=15,
    weight="bold"
)

plt.xlabel("Topik")
plt.ylabel("Jumlah Berita")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "8_topik_positif.png",
    dpi=300
)

plt.show()

positive_topic_df.to_csv(
    RESULT_DIR / "8_topik_positif.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✓ Topik Positif berhasil disimpan.")

# ==========================================
# TOPIK KELUHAN
# ==========================================

print("\nMenganalisis Topik Keluhan...")

negative_df = df[df["label"] == "Negatif"]

negative_topic_df = analyze_topics(negative_df)

plt.figure(figsize=(12,7))

plt.bar(
    negative_topic_df["Topik"],
    negative_topic_df["Jumlah Berita"],
    color="tomato"
)

plt.title(
    "Topik Keluhan pada Berita Bersentimen Negatif",
    fontsize=15,
    weight="bold"
)

plt.xlabel("Topik")

plt.ylabel("Jumlah Berita")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "9_topik_keluhan.png",
    dpi=300
)

plt.show()

negative_topic_df.to_csv(
    RESULT_DIR / "9_topik_keluhan.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✓ Topik Keluhan berhasil disimpan.")