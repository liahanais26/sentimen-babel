"""
utils.py
Utility functions untuk preprocessing teks Bahasa Indonesia.
"""

import re
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download tokenizer (hanya sekali)
nltk.download("punkt", quiet=True)

# Inisialisasi stemmer
stemmer = StemmerFactory().create_stemmer()

# Inisialisasi stopword
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())


def clean_text(text: str) -> str:
    """
    Membersihkan teks.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str):
    """
    Tokenisasi.
    """
    return text.split()


def remove_stopwords(tokens):
    """
    Menghapus stopword.
    """

    return [
        token
        for token in tokens
        if token not in stopwords
    ]


def stemming(tokens):
    """
    Stemming.
    """

    return [
        stemmer.stem(token)
        for token in tokens
    ]


def join_tokens(tokens):
    """
    Menggabungkan token menjadi kalimat.
    """

    return " ".join(tokens)