# NLP Project — Emotion classification of Reddit comments (GoEmotions)

Team: `<team name>` · Members: `<name – roll no>` (2–5 members)

## Topic & dataset
- **Dataset:** GoEmotions (Demszky et al., 2020) — 58k English Reddit comments labelled with 27 emotions + neutral.
- **Source:** https://huggingface.co/datasets/google-research-datasets/go_emotions (original files: https://github.com/google-research/google-research/tree/master/goemotions)
- **Licence:** Apache 2.0 (per the dataset card).
- **Our subset:** comments with exactly one emotion label → 45,446 comments, 28 classes (train/dev/test = official splits). After cleaning, 45,295 remain (151 became empty).
- **Target (Phase 3):** the emotion label.
- Raw data is **not committed**; `src/data_loader.py` downloads it from the official GitHub repo into `data/raw/`.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
python -m spacy download en_core_web_sm                      # used only for the tokenizer comparison
```
NLTK data (punkt, stopwords, wordnet, POS tagger) is downloaded automatically on first run.

## Run (reproduces everything)
```bash
python -m src.run_phase1      # Assignment 1: download -> clean -> data/processed/goemotions_clean.csv  (~30 s)
python -m src.run_phase2      # Assignment 2: TF-IDF vectorizers + CBOW / Skip-gram / fastText -> models/ (~1 min)
jupyter notebook notebooks/   # 01_phase1_cleaning.ipynb, 02_phase2_representations.ipynb (executed copies included)
```

## Layout
| Path | Purpose |
|---|---|
| `src/data_loader.py` | download/load GoEmotions |
| `src/patterns.py` | all regular expressions, documented |
| `src/cleaning.py` | the single cleaning pipeline (`clean_text`) |
| `src/representations.py` | BoW / TF-IDF / n-grams, Word2Vec, fastText helpers |
| `src/vectorize.py` | `vectorize(raw_doc, method)` — interface for Phase 3 |
| `notebooks/` | exploration, tokenization, cleaning (A1); representations and probing (A2) |
| `reports/` | stats JSON written by the scripts; PDF reports |
| `figures/` | plots used in the reports |

## Phase 3 interface
```python
from src.vectorize import vectorize
vectorize("I'm sooo happy today!!", method="fasttext")   # -> 100-d numpy vector
# methods: tfidf, tfidf_ngram (sparse rows); cbow, skipgram, fasttext (mean word vector)
```

## Design decisions (short)
- Emotion words and negations matter: stop-word list keeps `not/no/never/very/too/so/...`; contractions are expanded **before** stop-word removal.
- ALL-CAPS and `!` counts are saved as features before lower-casing.
- Lemmatisation (not stemming) so embedding neighbours stay readable.
- Phase 2 models are trained on the **train split only**; dev/test stay untouched for Phase 3.
- Corpus is small (~245k tokens after cleaning) — embedding results are interpreted cautiously in the report.

## Submission tags
`git tag assignment-1` (Phase 1) and `git tag assignment-2` (Phase 2), pushed to the remote.

## AI assistance disclosure
Claude (Anthropic) was used to help scaffold the code and notebooks in this repository. Every team member must be able to explain every line (see course academic-integrity policy). `<Edit this paragraph to describe exactly what your team used it for and what you changed.>`
