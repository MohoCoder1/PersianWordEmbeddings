# -*- coding: utf-8 -*-
"""Utility functions for text processing and similarity computation."""
import math

symbols = "?><؛،*#.,;:!؟«»()[]ــ”“\"،"


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    for s in symbols:
        text = text.replace(s, "")
    return text.strip().split()

def cosine_similarity_sparse(row1, row2) -> float:
    dot = row1.dot(row2.T)[0, 0]
    norm1 = math.sqrt(row1.multiply(row1).sum())
    norm2 = math.sqrt(row2.multiply(row2).sum())
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)