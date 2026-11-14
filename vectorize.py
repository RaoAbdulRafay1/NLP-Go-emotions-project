"""Phase 3 interface: raw document -> vector under a chosen representation.

    from src.vectorize import vectorize
    v = vectorize("I'm sooo happy today!!!", method="fasttext")

Methods
    tfidf        sparse 1 x V row  (unigram TF-IDF, min_df=2)
    tfidf_ngram  sparse 1 x V row  (uni+bigram TF-IDF, min_df=2)
    cbow         dense 100-d  (mean of Word2Vec-CBOW word vectors)
    skipgram     dense 100-d  (mean of Word2Vec-Skip-gram word vectors)
    fasttext     dense 100-d  (mean of fastText word vectors; handles OOV)

Models are only LOADED here (once, then cached) - nothing is ever retrained.
If a model file is missing, run  `python -m src.run_phase2`  once.
"""
from functools import lru_cache
import importlib
import re

import joblib
import numpy as np
from gensim.models import FastText, Word2Vec

from .representations import MODEL_DIR

METHODS = ("tfidf", "tfidf_ngram", "cbow", "skipgram", "fasttext")

_FILES = {
    "tfidf": MODEL_DIR / "tfidf_unigram.joblib",
    "tfidf_ngram": MODEL_DIR / "tfidf_ngram12.joblib",
    "cbow": MODEL_DIR / "w2v_cbow.model",
    "skipgram": MODEL_DIR / "w2v_skipgram.model",
    "fasttext": MODEL_DIR / "fasttext.model",
}

# Text cleaning must match Assignment 1. We try the team's Assignment-1 cleaner
# first; the built-in fallback only approximates it (see README).
_A1_CANDIDATES = [("src.preprocess", "clean_text"), ("src.cleaning", "clean_text"),
                  ("src.clean", "clean_text"), ("src.preprocessing", "clean_text"),
                  ("src.preprocess", "preprocess"), ("src.cleaning", "clean")]


def _find_assignment1_cleaner():
    for mod, fn in _A1_CANDIDATES:
        try:
            f = getattr(importlib.import_module(mod), fn, None)
            if callable(f):
                return f
        except Exception:
            continue
    return None


_NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "cannot"}
_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "is", "are", "was",
         "were", "be", "been", "being", "am", "i", "me", "my", "you", "your", "it", "its", "this", "that",
         "these", "those", "do", "does", "did", "have", "has", "had", "as", "by", "from", "up", "out",
         "than", "then", "too", "very", "just", "s", "t", "we", "they", "them", "our", "us"}


def _fallback_clean(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"u/\w+|r/\w+|@\w+", " ", t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'(?:m|s|re|ve|ll|d)\b", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(w for w in t.split() if w in _NEGATIONS or w not in _STOP)


def clean(raw: str) -> str:
    """Raw comment -> space-joined clean tokens (same format as `text_clean`)."""
    f = _find_assignment1_cleaner()
    return f(raw) if f else _fallback_clean(raw)


@lru_cache(maxsize=None)
def _load(method: str):
    """Load a saved model/vectorizer once. Never trains."""
    if method not in _FILES:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    path = _FILES[method]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.run_phase2` once")
    if method.startswith("tfidf"):
        return joblib.load(path)
    return (FastText if method == "fasttext" else Word2Vec).load(str(path))


def vectorize(raw_document: str, method: str = "fasttext"):
    """Return the vector of one RAW document under `method`.

    tfidf / tfidf_ngram -> scipy csr_matrix, shape (1, n_features)
    cbow / skipgram / fasttext -> numpy array, shape (100,)
        all-zero if no usable token remains; Word2Vec ignores OOV tokens,
        fastText builds a vector for them from character n-grams.
    """
    model = _load(method)
    cleaned = clean(raw_document)
    if method.startswith("tfidf"):
        return model.transform([cleaned])
    tokens = cleaned.split()
    if method != "fasttext":
        tokens = [t for t in tokens if t in model.wv.key_to_index]
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean([model.wv[t] for t in tokens], axis=0)


def vectorize_many(raw_documents, method: str = "fasttext"):
    """Stack vectors for a list of documents (Phase 3 helper)."""
    from scipy.sparse import vstack
    vecs = [vectorize(d, method) for d in raw_documents]
    return vstack(vecs) if method.startswith("tfidf") else np.vstack(vecs)
