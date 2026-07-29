"""
06_evaluation.py

Tahapan:
1. Membaca dataset
2. Train-test split
3. Memuat model dan TF-IDF
4. Prediksi
5. Evaluasi
6. Menyimpan classification report
7. Menyimpan confusion matrix
8. Visualisasi performa model
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LABELED_DATA,
    RESULT_DIR,
    SVM_MODEL,
    TFIDF_MODEL,
    RANDOM_STATE,
    TEST_SIZE,
    CLASSIFICATION_REPORT,
    CONFUSION_MATRIX,
)


def timer(start):
    return f"{time.time()-start:.2f} detik"


def main():

    total_start = time.time()

    print("=" * 60)
    print("06. EVALUASI MODEL SVM")
    print("=" * 60)

    # =====================================================
    # Membaca Dataset
    # =====================================================

    if not LABELED_DATA.exists():
        raise FileNotFoundError(LABELED_DATA)

    print("\nMembaca dataset...")

    df = pd.read_excel(LABELED_DATA)

    print(f"Jumlah data awal : {len(df)}")

    # =====================================================
    # Filter hanya Positif dan Negatif
    # =====================================================

    df = df[df["label"].isin(["Positif", "Negatif"])].copy()

    print(f"Jumlah data setelah filter : {len(df)}")

    print("\nDistribusi Label")
    print(df["label"].value_counts())

    # =====================================================
    # Split
    # =====================================================

    print("\nMembagi data train dan test...")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_final"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print(f"\nData train : {len(X_train)}")
    print(f"Data test  : {len(X_test)}")

    print("\nDistribusi Label Train")
    print(y_train.value_counts())

    print("\nDistribusi Label Test")
    print(y_test.value_counts())

    # =====================================================
    # Load Model
    # =====================================================

    print("\nMemuat model...")

    model = joblib.load(SVM_MODEL)
    vectorizer = joblib.load(TFIDF_MODEL)

    X_test_tfidf = vectorizer.transform(X_test)

    # =====================================================
    # Prediksi
    # =====================================================

    print("\nMelakukan prediksi...")

    start = time.time()

    y_pred = model.predict(X_test_tfidf)

    print(f"Selesai ({timer(start)})")

    # =====================================================
    # Evaluasi
    # =====================================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Positif",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Positif",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Positif",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("HASIL EVALUASI")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification Report")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    # =====================================================
    # Classification Report
    # =====================================================

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(CLASSIFICATION_REPORT)

    print("\nClassification Report berhasil disimpan.")

    # =====================================================
    # Confusion Matrix
    # =====================================================

    plt.figure(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Negatif", "Positif"],
        cmap="Blues",
        values_format="d",
    )

    plt.title("Confusion Matrix Model SVM")

    plt.tight_layout(pad=2)

    plt.savefig(
        CONFUSION_MATRIX,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Confusion Matrix berhasil disimpan.")

    # =====================================================
    # Visualisasi Performa
    # =====================================================

    metrics = {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100,
    }

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        metrics.keys(),
        metrics.values(),
        width=0.55,
        edgecolor="black",
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            f"{height:.2f}%",
            ha="center",
            fontweight="bold",
        )

    plt.title("Performa Model SVM")

    plt.ylabel("Persentase (%)")

    plt.ylim(0, 100)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "9_performa_model.png",
        dpi=300,
    )

    plt.close()

    print("Visualisasi performa berhasil disimpan.")

    print("\nLokasi hasil:")

    print(CLASSIFICATION_REPORT)

    print(CONFUSION_MATRIX)

    print(RESULT_DIR / "9_performa_model.png")

    print(f"\nTotal waktu : {timer(total_start)}")


if __name__ == "__main__":
    main()