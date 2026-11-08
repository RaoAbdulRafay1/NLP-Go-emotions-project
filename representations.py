"""Phase 2 building blocks: count-based representations and learned embeddings."""
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "processed" / "goemotions_clean.csv"
MODEL_DIR = ROOT / "models"

# One shared hyper-parameter set so CBOW / Skip-gram / fastText are comparable.
W2V_PARAMS = dict(vector_size=100, window=5, min_count=2, epochs=30,
                  negative=5, workers=1, seed=42)   # workers=1 -> reproducible


def load_clean(split: str = "train") -> pd.DataFrame:
    """Load the Assignment-1 output. Phase 2 uses the TRAIN split only, so that
    nothing learned here leaks information from dev/test into Phase 3."""
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False)
    return df[df["split"] == split].reset_index(drop=True) if split else df


def sparsity(matrix) -> float:
    """% of zero entries in a (sparse) matrix."""
    total = matrix.shape[0] * matrix.shape[1]
    return 100.0 * (1 - matrix.nnz / total)


# ---- Part A -------------------------------------------------------------------
def build_bow(docs, ngram_range=(1, 1), min_df=1):
    vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    return vec, vec.fit_transform(docs)


def build_tfidf(docs, ngram_range=(1, 1), min_df=1):
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    return vec, vec.fit_transform(docs)


def ngram_growth_table(docs, min_dfs=(1, 2)) -> pd.DataFrame:
    rows = []
    for min_df in min_dfs:
        for name, rng in [("unigrams", (1, 1)), ("uni+bigrams", (1, 2)),
                          ("uni+bi+trigrams", (1, 3))]:
            _, X = build_bow(docs, rng, min_df)
            rows.append({"min_df": min_df, "n-gram range": name,
                         "features": X.shape[1], "non-zeros": X.nnz,
                         "sparsity %": round(sparsity(X), 4)})
    return pd.DataFrame(rows)


# ---- Part B -------------------------------------------------------------------
def tokenised(docs):
    """Cleaned docs are space-joined tokens, so split() restores the token lists."""
    return [d.split() for d in docs]


def train_word2vec(sentences, sg: int, **overrides) -> Word2Vec:
    params = {**W2V_PARAMS, **overrides}
    return Word2Vec(sentences=sentences, sg=sg, **params)


def train_fasttext(sentences, sg: int = 1, **overrides) -> FastText:
    params = {**W2V_PARAMS, **overrides}
    return FastText(sentences=sentences, sg=sg, min_n=3, max_n=6,
                    bucket=100_000, **params)   # small hash table: 40 MB, not 800 MB


# ---- Part C -------------------------------------------------------------------
def neighbours(model, word, topn=5):
    wv = model.wv
    if word not in wv.key_to_index and not isinstance(model, FastText):
        return None                      # Word2Vec cannot answer for OOV words
    return wv.most_similar(word, topn=topn)


def cosine(model, a, b):
    return float(model.wv.similarity(a, b))
