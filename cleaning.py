"""Single, reusable cleaning pipeline for GoEmotions (Assignment 1, Task 4).

ORDER OF STEPS (and why):
  0. Features from the RAW text (ALL-CAPS words, '!' count) - must happen before
     case folding / punctuation removal destroys the signal.
  1. Unicode normalisation (NFC) + curly quotes -> ASCII. Must be first so every
     later regex sees one canonical form of each character.
  2. Noise removal: placeholders, subreddit/user mentions, URLs/e-mails, HTML
     entities, markdown. Emojis are converted to words (they carry emotion).
  3. Contraction expansion ("didn't" -> "did not"): done BEFORE stop-word removal
     so negations survive as the token "not".
  4. Elongation collapse ("sooooo" -> "soo").
  5. Case folding (lower-case). Safe now because step 0 saved the caps signal.
  6. Tokenisation (NLTK word_tokenize) and drop tokens with no letters.
  7. POS tagging on the full sentence (the tagger needs context).
  8. Stop-word removal with a CUSTOM list that keeps negations/intensifiers.
  9. Lemmatisation (WordNet, POS-aware) - chosen over stemming because it returns
     real words ("happiness" stays, "feeling" -> "feel"), which keeps embeddings
     in Assignment 2 interpretable.
"""
import re
import unicodedata

import emoji
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from . import patterns as P

# ---- resources --------------------------------------------------------------
for _pkg, _path in [("punkt_tab", "tokenizers/punkt_tab"),
                    ("stopwords", "corpora/stopwords"),
                    ("wordnet", "corpora/wordnet"),
                    ("averaged_perceptron_tagger_eng",
                     "taggers/averaged_perceptron_tagger_eng")]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ---- stop words: keep words that change/intensify emotion ---------------------
KEEP_WORDS = {
    # negations - "not happy" must not become "happy"
    "no", "nor", "not", "never", "neither", "without",
    # intensifiers / contrast that shift emotional strength
    "very", "too", "so", "most", "more", "but", "against",
}
STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
# tokens left over by contraction fragments (e.g. "don", "t", "s")
STOP_WORDS |= {"s", "t", "d", "ll", "m", "re", "ve", "y"}
STOP_WORDS -= KEEP_WORDS

# ---- contractions ---------------------------------------------------------------
_CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bshan't\b", "shall not"),
    (r"\bain't\b", "is not"), (r"\blet's\b", "let us"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"),
    (r"'m\b", " am"), (r"'d\b", " would"), (r"'s\b", ""),
]
_CONTRACTIONS = [(re.compile(p, re.I), r) for p, r in _CONTRACTIONS]

_QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'",
                            "\u201c": '"', "\u201d": '"', "\u2033": '"'})

_lemmatizer = WordNetLemmatizer()


# ---- step functions ---------------------------------------------------------------
def extract_raw_features(text: str) -> dict:
    """Step 0: signals that cleaning would otherwise destroy."""
    return {
        "n_caps_words": len(P.ALLCAPS_WORD.findall(text)),
        "n_exclaim": len(P.EXCLAIM.findall(text)),
        "has_punct_run": int(bool(P.PUNCT_RUN.search(text))),
    }


def normalize_unicode(text: str) -> str:
    """Step 1: NFC + curly quotes -> straight quotes."""
    return unicodedata.normalize("NFC", text).translate(_QUOTE_MAP)


def remove_noise(text: str) -> str:
    """Step 2: remove artefacts, keep emotional content (emojis -> words)."""
    text = P.PLACEHOLDER.sub(" ", text)
    text = P.SUBREDDIT_USER.sub(" ", text)
    text = P.URL.sub(" ", text)
    text = P.EMAIL.sub(" ", text)
    text = P.HTML_ENTITY.sub(" ", text)
    text = P.MARKDOWN.sub(" ", text)
    # "😘" -> " face blowing a kiss "
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")
    return text


def expand_contractions(text: str) -> str:
    """Step 3."""
    for pat, rep in _CONTRACTIONS:
        text = pat.sub(rep, text)
    return text


def collapse_elongation(text: str) -> str:
    """Step 4: 'sooooo' -> 'soo' (keeps 'good', 'too' untouched)."""
    return P.ELONGATION.sub(r"\1\1", text)


def _wn_pos(tag: str) -> str:
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def clean_tokens(text: str, remove_stopwords: bool = True,
                 lemmatize: bool = True) -> list:
    """Run steps 1-9 and return the final token list."""
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = expand_contractions(text)
    text = collapse_elongation(text)
    text = text.lower()                                   # step 5
    tokens = [t for t in word_tokenize(text) if re.search(r"[a-z]", t)]   # step 6
    tagged = nltk.pos_tag(tokens)                         # step 7
    out = []
    for tok, tag in tagged:
        if remove_stopwords and tok in STOP_WORDS:        # step 8
            continue
        tok = tok.strip("'-.")                            # stray punctuation
        if not tok:
            continue
        if lemmatize:                                     # step 9
            tok = _lemmatizer.lemmatize(tok, _wn_pos(tag))
        out.append(tok)
    return out


def clean_text(text: str, **kwargs) -> str:
    """THE cleaning function: raw comment -> cleaned string (space-joined tokens)."""
    return " ".join(clean_tokens(text, **kwargs))


def raw_tokens(text: str) -> list:
    """Baseline tokens (no cleaning) used for the vocabulary-BEFORE number."""
    return re.findall(r"\w+|[^\w\s]", text)
