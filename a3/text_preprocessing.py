from collections import Counter

import nltk
import numpy as np
from nltk.corpus import reuters, stopwords


class ReuterDataTokenizer:
    def __init__(
        self,
        corpus_size,
        common_threshold: float,
        rare_threshold: int,
    ):
        self.docs, self.stop_words = load_data(corpus_size)
        self.common_threshold = common_threshold
        self.rare_threshold = rare_threshold

    def preprocess(self) -> list[list[str]]:
        samples = gen.choice(self.docs, replace=False)
        docs_word_split: list[list[str]] = [text.lower().split() for text in samples]
        docs_word_split = [
            [word for word in self.docs if word not in self.stop_words]
            for doc in self.docs
        ]
        cleaned_docs = remove_common_and_rare_words(
            docs_word_split, self.common_threshold, self.rare_threshold
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
    }
    cleaned_docs = [[word for word in doc if word in keep] for doc in docs]
    return cleaned_docs


def load_data(sample_size) -> tuple[list[str], set[str]]:
    file_ids = reuters.fileids()
    docs: list[str] = [reuters.raw(id) for id in file_ids]
    stop_words = set(stopwords.words("english"))
    return docs, stop_words


def main():
    if DOWNLOAD_DATA:
        nltk.download("reuters")
        nltk.download("stopwords")

    sample_size = 1000
    common_threshold = 0.6
    rare_threshold = 15

    tokenizer = ReuterDataTokenizer(sample_size, common_threshold, rare_threshold)
    tokenizer.preprocess()


if __name__ == "__main__":
    DOWNLOAD_DATA = False
    gen = np.random.default_rng(seed=42)
    main()
