import pickle
from collections import Counter

import numpy as np
from text_preprocessing import BOWDataTokenizer


class LDA:
    def __init__(self, corpus, tokenizer, n_topics, alpha, beta):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.tokenizer = tokenizer
        self.z = None
        self.phi = None
        self.theta = None
        self.corpus = corpus

    def train(
        self,
        iterations: int,
        rng: np.random.Generator,
    ):
        V = len(self.tokenizer)
        D = len(self.corpus)
        N = sum([len(text) for text in self.corpus])

        topics = np.arange(0, self.n_topics, dtype=np.int64)

        self.z = rng.integers(low=0, high=self.n_topics, size=N)
        w_ids = np.array(
            [self.tokenizer(word) for text in self.corpus for word in text]
        )
        d_ids = [d for d, text in enumerate(self.corpus) for _ in text]

        n_dk = np.zeros((D, self.n_topics), dtype=np.int64)
        n_kw = np.zeros((self.n_topics, V), dtype=np.int64)
        n_k = np.zeros(self.n_topics, dtype=np.int64)

        np.add.at(n_dk, (d_ids, self.z), 1)
        np.add.at(n_kw, (self.z, w_ids), 1)
        np.add.at(n_k, self.z, 1)

        for iteration in range(iterations):
            print(f"Starting iteration {iteration}")
            for i, id in enumerate(w_ids):
                old_z = self.z[i]
                n_dk[d_ids[i], old_z] -= 1
                n_kw[old_z, id] -= 1
                n_k[old_z] -= 1

                q = np.array(
                    [
                        (
                            (n_dk[d_ids[i], k] + self.alpha)
                            * (n_kw[k, id] + self.beta)
                            / (n_k[k] + V * self.beta)
                        )
                        for k in range(self.n_topics)
                    ]
                )
                p = q / np.sum(q)
                new_z = rng.choice(
                    topics,
                    p=p,
                )

                n_dk[d_ids[i], new_z] += 1
                n_kw[new_z, id] += 1
                n_k[new_z] += 1

                self.z[i] = new_z

        N_d = np.sum(n_dk, axis=1)

        self.theta = (n_dk + self.alpha) / (N_d[:, None] + self.n_topics * self.alpha)
        self.phi = (n_kw + self.beta) / (n_k[:, None] + V * self.beta)

    def top_words_relative(self, n_top: int = 20):
        words = np.array([w for text in self.tokenizer.corpus for w in text])
        total_counter = Counter(words)

        top_words = []
        for t in range(self.n_topics):
            counter = Counter(words[self.z == t])
            relative = {w: c / total_counter[w] for w, c in counter.items()}
            top_words.append(sorted(relative, key=relative.get, reverse=True)[:n_top])
        return top_words

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(save_path):
        with open(save_path, "rb") as f:
            model = pickle.load(f)
        return model


def print_k_best_words(top_k_best, umass_score) -> None:
    for topic, words in enumerate(top_k_best):
        print(f"Topic {topic}: \n")
        for placement, w in enumerate(words):
            print(f"{placement}. {w}")
        print(f"C_Umass score:{umass_score[topic]}")
        print()


def D_one(w: str, corpus) -> int:
    counter = 0
    for text in corpus:
        if w in text:
            counter += 1
    return counter


def D_two(w1: str, w2: str, corpus) -> int:
    counter = 0
    for text in corpus:
        if (w1 in text) & (w2 in text):
            counter += 1
    return counter


def umas_coherence(words: list[list[str]], corpus):
    n_topics = len(words)
    c_umass = np.zeros(n_topics)
    for k in range(n_topics):
        topic_words = words[k]
        n_words = len(topic_words)
        for m in range(1, n_words):
            for l in range(m):
                c_umass[k] += np.log(
                    (D_two(topic_words[m], topic_words[l], corpus) + 1)
                    / D_one(topic_words[l], corpus)
                )
    return c_umass


def main():
    tokenizer = BOWDataTokenizer.load("./tokenizer.pkl")
    model_path = "./lda_model.pkl"

    if LOAD_MODEL:
        model = LDA.load(model_path)

    else:
        n_topics = 10
        alpha = 0.01
        beta = 0.01
        iterations = 200
        model = LDA(tokenizer.corpus, tokenizer, n_topics, alpha, beta)
        gen = np.random.default_rng(seed=42)
        model.train(iterations, gen)
        model.save(model_path)

    k = 20
    words = model.top_words_relative(k)

    umass_score = umas_coherence(words, tokenizer.corpus)
    print_k_best_words(words, umass_score)


if __name__ == "__main__":
    LOAD_MODEL = True
    main()
