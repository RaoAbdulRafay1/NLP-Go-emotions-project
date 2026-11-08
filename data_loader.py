"""Load the GoEmotions dataset (Demszky et al., 2020; Apache-2.0).

The raw files are downloaded from the official GitHub repository on first use
and cached in data/raw/ (git-ignored: we never commit raw data).
"""
from pathlib import Path
import urllib.request

import pandas as pd

BASE_URL = ("https://raw.githubusercontent.com/google-research/"
            "google-research/master/goemotions/data/")
FILES = ["train.tsv", "dev.tsv", "test.tsv", "emotions.txt"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_raw(raw_dir: Path = RAW_DIR) -> None:
    """Download the official split files if they are not cached yet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = raw_dir / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, dest)


def load_goemotions(single_label_only: bool = True,
                    raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return a DataFrame with columns: id, text, label, split.

    GoEmotions is multi-label (a comment can carry several emotions). For the
    classification phases we keep only comments with exactly ONE emotion so
    the target is a clean multi-class label (84% of the data survives).
    """
    download_raw(raw_dir)
    emotions = [e.strip() for e in (raw_dir / "emotions.txt").read_text().splitlines()]
    frames = []
    for split, fname in [("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")]:
        df = pd.read_csv(raw_dir / fname, sep="\t", header=None,
                         names=["text", "label_ids", "id"], dtype=str,
                         keep_default_na=False)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["n_labels"] = df["label_ids"].str.count(",") + 1
    if single_label_only:
        df = df[df["n_labels"] == 1].copy()
    df["label"] = df["label_ids"].str.split(",").str[0].astype(int).map(dict(enumerate(emotions)))
    return df[["id", "text", "label", "split"]].reset_index(drop=True)


if __name__ == "__main__":
    d = load_goemotions()
    print(d.shape)
    print(d["label"].value_counts().head())
