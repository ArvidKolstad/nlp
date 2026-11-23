import pickle
import re
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import reuters, stopwords

# Arvid Kolstad, Simon Eriksson, Hassan Abla, Filip Björnåsen

TOKEN_RE = re.compile(r"[a-z]+")


class BOWDataTokenizer:
    def __init__(
        self, vocabular: dict[str, int], total_words_corpus, corpus: list[list[str]]
    ):
        self.vocabular: dict[str, int] = vocabular
        self.inverse_vocabular: dict[int, str] = {v: k for k, v in vocabular.items()}
        self.N: int = total_words_corpus
        self.corpus: list[list[str]] = corpus

    def __call__(self, word: str) -> int:
        return self.vocabular[word]

    def inverse(self, index) -> str:
        return self.inverse_vocabular[index]

    def get_index_array(self, words: list[str]) -> np.ndarray:
        return np.array([self(w) for w in words])

    def get_word_list(self, indexes: np.ndarray) -> list[str]:
        return [self.inverse(i) for i in indexes]

    def __len__(self):
        return len(self.vocabular)

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(save_path):
        with open(save_path, "rb") as f:
            model = pickle.load(f)
        return model


def preprocess(
    docs,
    stop_words,
    corpus_size,
    common_threshold,
    rare_threshold,
    rng,
) -> list[list[str]]:

    samples = rng.choice(docs, size=corpus_size, replace=False)
    docs_word_split: list[list[str]] = [
        TOKEN_RE.findall(text.lower()) for text in samples
    ]

    docs_word_split = [
        [word for word in doc if word not in stop_words] for doc in docs_word_split
    ]

    cleaned_docs = remove_common_and_rare_words(
        docs_word_split, common_threshold, rare_threshold
    )

    return cleaned_docs


def remove_common_and_rare_words(
    docs: list[list[str]], common_threshold: float = 0.6, rare_threshold: int = 10
):
    total_docs = len(docs)
    doc_freq = Counter(tok for doc in docs for tok in set(doc))
    total_counts = Counter(tok for doc in docs for tok in doc)

    keep = {
        word
        for word, count in total_counts.items()
        if (count >= rare_threshold)
        and (doc_freq[word] / total_docs < common_threshold)
        and len(word) >= 3
    }
    cleaned_docs = [[word for word in doc if word in keep] for doc in docs]

    return cleaned_docs


def get_vocabulizer(corpus_size, common_threshold, rare_threshold, rng):
    docs, stop_words = load_data()
    cleaned_docs = preprocess(
        docs, stop_words, corpus_size, common_threshold, rare_threshold, rng
    )
    unique_words = sorted({w for doc in cleaned_docs for w in doc})
    vocabular = {w: idx for idx, w in enumerate(unique_words)}
    total_words_corpus = np.sum([len(words) for words in cleaned_docs])
    print(f"Total words in corpus: {total_words_corpus}")

    return BOWDataTokenizer(vocabular, total_words_corpus, cleaned_docs)


def load_data() -> tuple[list[str], set[str]]:
    file_ids = reuters.fileids()
    docs: list[str] = [reuters.raw(id) for id in file_ids]
    stop_words = set(stopwords.words("english"))
    return docs, stop_words


def main():
    tokenizer_path = "tokenizer.pkl"
    if DOWNLOAD_DATA:
        nltk.download("reuters")
        nltk.download("stopwords")
    if LOAD_TOKENIZER:
        tokenizer = BOWDataTokenizer.load(tokenizer_path)
    else:
        sample_size = 5000
        common_threshold = 0.6
        rare_threshold = 10

        tokenizer = get_vocabulizer(sample_size, common_threshold, rare_threshold, gen)
        tokenizer.save(tokenizer_path)

    print(len(tokenizer))


if __name__ == "__main__":
    DOWNLOAD_DATA = False
    LOAD_TOKENIZER = False
    gen = np.random.default_rng(seed=42)
    main()
