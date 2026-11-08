"""Regular expressions designed for GoEmotions (informal Reddit comments).

Each pattern is explained in the report (Task 3). All are compiled once here
so the cleaning pipeline and the notebooks use exactly the same definitions.
"""
import re

# 1. Anonymisation placeholders inserted by the dataset authors: [NAME], [RELIGION].
#    They are artefacts, not words the author wrote, and carry no emotion.
PLACEHOLDER = re.compile(r"\[(?:NAME|RELIGION)\]")

# 2. Reddit mentions: r/hockey, /r/hockey, u/someone, /u/someone.
#    (?<!\w) stops us matching inside words such as "chair/user".
SUBREDDIT_USER = re.compile(r"(?<!\w)/?(?:r|u)/\w+")

# 3. Character elongation: "sooooo", "nooo", "loooove" (3+ same char in a row).
#    Back-reference \1 repeats the captured char; we collapse the run to 2.
ELONGATION = re.compile(r"(\w)\1{2,}")

# 4. URLs and e-mail addresses (defensive: GoEmotions has none after
#    anonymisation, which exploration confirms with 0 matches).
URL = re.compile(r"(?:https?://|www\.)\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# 5. ALL-CAPS words (3+ letters) - a shouting signal we keep as a feature
#    BEFORE lower-casing destroys it.
ALLCAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")

# 6. Exclamation marks / runs of ! or ? (emphasis, disbelief) - counted as features.
EXCLAIM = re.compile(r"!")
PUNCT_RUN = re.compile(r"[!?]{2,}")

# 7. Markdown emphasis left over from Reddit: **bold**, __x__, ~~strike~~, `code`.
MARKDOWN = re.compile(r"\*+|_{2,}|~~|`")

# 8. HTML entities such as &amp; &gt; &#x200B;
HTML_ENTITY = re.compile(r"&(?:#x?[0-9a-fA-F]+|\w+);")

# 9. Digits / prices / years (explored, then dropped by the cleaner).
NUMBER = re.compile(r"\$?\d[\d,.]*")

# Extra whitespace
WHITESPACE = re.compile(r"\s+")
