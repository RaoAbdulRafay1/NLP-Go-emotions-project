"""Phase-3 interface: raw document -> vector under a chosen representation.

    from src.vectorize import vectorize
    v = vectorize("I am sooo happy today!!", method="fasttext")

Models are loaded ONCE from models/ (never retrained here).
Methods: "bow" (not saved - use tfidf), "tfidf", "cbow", "skipgram", "fasttext".
"""
from functools import lru_cache

import joblib
import numpy as np
from gensim.models import FastText, Word2Vec

from .cleaning import clean_text
from .representations import MODEL_DIR

METHODS = ("tfidf", "tfidf_ngram", "cbow", "skipgram", "fasttext")


@lru_cache(maxsize=None)
def _load(method: str):
    if method == "tfidf":
        return joblib.load(MODEL_DIR / "tfidf_unigram.joblib")
    if method == "tfidf_ngram":
        return joblib.load(MODEL_DIR / "tfidf_ngram12.joblib")
    if method == "cbow":
        return Word2Vec.load(str(MODEL_DIR / "w2v_cbow.model"))
    if method == "skipgram":
        return Word2Vec.load(str(MODEL_DIR / "w2v_skipgram.model"))
    if method == "fasttext":
        return FastText.load(str(MODEL_DIR / "fasttext.model"))
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def vectorize(raw_doc: str, method: str = "tfidf"):
    """Return the vector of ONE raw document.

    tfidf / tfidf_ngram -> scipy sparse row (1 x vocab)
    cbow / skipgram / fasttext -> dense numpy array (100,), the mean of the
        word vectors of the cleaned tokens. Words unknown to Word2Vec are
        skipped; fastText builds vectors for them from character n-grams.
        A document with no usable token returns the zero vector.
    """
    cleaned = clean_text(raw_doc)
    model = _load(method)
    if method in ("tfidf", "tfidf_ngram"):
        return model.transform([cleaned])
    tokens = cleaned.split()
    if method == "fasttext":
        vecs = [model.wv[t] for t in tokens]
    else:
        vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)
