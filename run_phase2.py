"""Phase 2 entry point: build + SAVE every representation.  python -m src.run_phase2"""
import json

import joblib

from . import representations as R


def main():
    R.MODEL_DIR.mkdir(exist_ok=True)
    (R.ROOT / "reports").mkdir(exist_ok=True)
    train = R.load_clean("train")
    docs = train["text_clean"].tolist()
    print(f"{len(docs):,} training documents")

    # Part A: count-based. Vectorizers are saved for Phase 3.
    vec_u, X_u = R.build_tfidf(docs, (1, 1), min_df=2)
    vec_n, X_n = R.build_tfidf(docs, (1, 2), min_df=2)
    joblib.dump(vec_u, R.MODEL_DIR / "tfidf_unigram.joblib")
    joblib.dump(vec_n, R.MODEL_DIR / "tfidf_ngram12.joblib")
    print("tfidf unigram", X_u.shape, "| uni+bigram", X_n.shape)

    # Part B: embeddings, identical hyper-parameters
    sents = R.tokenised(docs)
    cbow = R.train_word2vec(sents, sg=0)
    skip = R.train_word2vec(sents, sg=1)
    ft = R.train_fasttext(sents, sg=1)
    cbow.save(str(R.MODEL_DIR / "w2v_cbow.model"))
    skip.save(str(R.MODEL_DIR / "w2v_skipgram.model"))
    ft.save(str(R.MODEL_DIR / "fasttext.model"))
    print("vocab  cbow/skip/fasttext:", len(cbow.wv), len(skip.wv), len(ft.wv))

    info = {"hyperparameters": R.W2V_PARAMS, "n_train_docs": len(docs),
            "n_tokens": sum(map(len, sents)), "w2v_vocab": len(cbow.wv)}
    (R.ROOT / "reports" / "phase2_stats.json").write_text(json.dumps(info, indent=2))
    print(info)


if __name__ == "__main__":
    main()
