# -*- coding: utf-8 -*-
"""
DataManager: encapsulates data loading, vocabulary building, co-occurrence matrix,
similarity computation and PCA preparation.
"""
from collections import Counter
from typing import List, Tuple
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import PCA
from datasets import load_dataset
import plotly.express as px
import pandas as pd
from core.utils import tokenize, cosine_similarity_sparse

DATASET_NAME = "QomSSLab/law-text-dataset-fa"
SPLITER = "train"

class DataManager:



    def __init__(self):
        self.texts: List[str] = []
        self.vocab: List[str] = []
        self.word_to_index = {}
        self.word_freq = Counter()
        self.matrix: csr_matrix = None

    def load_law_dataset(self, limit: int = 1000) -> List[str]:
        dataset = load_dataset(DATASET_NAME, split=SPLITER)
        texts = []
        if limit and limit > 0:
            for i, item in enumerate(dataset):
                if "text" in item:
                    texts.append(item["text"])
                if len(texts) >= limit:
                    break
        else:
            for item in dataset:
                if "text" in item:
                    texts.append(item["text"])
        self.texts = texts
        return texts

    def build_vocabulary(self, max_vocab_size: int = 2000) -> Tuple[List[str], dict]:
        wf = Counter()
        for t in self.texts:
            wf.update(tokenize(t))
        self.word_freq = wf
        most_common = [w for w, _ in wf.most_common(max_vocab_size)]
        self.vocab = sorted(most_common)
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        return self.vocab, self.word_to_index

    def build_cooccurrence_matrix(self, window_size: int = 1) -> csr_matrix:
        vocab_size = len(self.vocab)
        M = dok_matrix((vocab_size, vocab_size), dtype=float)
        for text in self.texts:
            words = tokenize(text)
            for i, word in enumerate(words):
                if word not in self.word_to_index:
                    continue
                ti = self.word_to_index[word]
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    cw = words[j]
                    if cw not in self.word_to_index:
                        continue
                    ci = self.word_to_index[cw]
                    M[ti, ci] += 1.0
        self.matrix = M.tocsr()
        return self.matrix

    def most_similar(self, target_word: str, top_n: int = 5):
        if target_word not in self.word_to_index:
            return []
        tidx = self.word_to_index[target_word]
        trow = self.matrix.getrow(tidx)
        sims = []
        for w in self.vocab:
            if w == target_word:
                continue
            vrow = self.matrix.getrow(self.word_to_index[w])
            s = cosine_similarity_sparse(trow, vrow)
            sims.append((w, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_n]

    def plot_pca_for_target(self, target_word: str, top_n: int = 20):
        """
        Plots PCA of the target word and its most similar words.
        """
        if self.matrix is None:
            raise RuntimeError("Matrix not built yet")

        if target_word not in self.word_to_index:
            raise ValueError(f"کلمه '{target_word}' در واژگان وجود ندارد.")

        # index of target word
        target_idx = self.word_to_index[target_word]

        # finding most similar words
        similar = self.most_similar(target_word, top_n=top_n)
        indices = [self.word_to_index[w] for w, _ in similar]

        # these are all the indices we need to plot
        all_indices = [target_idx] + indices
        if hasattr(self.matrix, "toarray"):
            X = self.matrix[all_indices, :].toarray()
        else:
            X = np.array(self.matrix[all_indices, :], dtype=float)

        # PCA
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)

        # prepare words with RTL support
        words = [target_word] + [w for w, _ in similar]
        words_fixed = []
        for w in words:
            # adding RTL mark
            words_fixed.append("\u202B" + w)

        # create dataframe for plotly
        df = pd.DataFrame({
            'کلمه': words_fixed,
            'x': X2[:, 0],
            'y': X2[:, 1]
        })

        fig = px.scatter(
            df,
            x='x',
            y='y',
            text='کلمه',
            title=f"PCA - کلمه هدف: {target_word}",
            width=900,
            height=700
        )

        fig.update_traces(textposition='top center', marker=dict(size=10, color='red'))
        fig.update_layout(
            title_font=dict(size=22, family="Vazirmatn, Tahoma"),
            font=dict(family="Vazirmatn, Tahoma", size=14, color="black"),
            xaxis_title="مولفه اول (PC1)",
            yaxis_title="مولفه دوم (PC2)",
        )

        fig.show()
