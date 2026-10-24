"""
Sentiment Analysis Utilities
Author : TejeswaraReddy-Dareddy
Date   : 2026-10-21
Updated: 2026-10-24
Description:
    NLP utilities for the BA-Project text mining pipeline.
    Covers text cleaning, TextBlob sentiment scoring, VADER scoring,
    batch processing, aggregation by group, and visualisation helpers.
"""

import re
import string
import warnings
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional NLTK / VADER import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# ---------------------------------------------------------------------------
# 1. Text Cleaning
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "this", "that", "was", "are",
    "with", "as", "by", "from", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "its", "their", "our", "your", "my",
}

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Clean raw text: lowercase, strip punctuation, collapse whitespace.

    Args:
        text             : Raw input string.
        remove_stopwords : If True, remove common English stopwords.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # mentions / hashtags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)                      # numbers
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in _STOPWORDS)
    return text


# ---------------------------------------------------------------------------
# 2. TextBlob Sentiment
# ---------------------------------------------------------------------------
def textblob_sentiment(text: str) -> Dict:
    """Return TextBlob polarity, subjectivity, and label.

    Returns:
        Dict with keys: polarity, subjectivity, label.
    """
    if not TEXTBLOB_AVAILABLE:
        return {"polarity": 0.0, "subjectivity": 0.0, "label": "neutral"}
    blob = TextBlob(str(text))
    p    = blob.sentiment.polarity
    s    = blob.sentiment.subjectivity
    label = "positive" if p > 0.05 else ("negative" if p < -0.05 else "neutral")
    return {"polarity": round(p, 4), "subjectivity": round(s, 4), "label": label}


# ---------------------------------------------------------------------------
# 3. VADER Sentiment
# ---------------------------------------------------------------------------
def vader_sentiment(text: str) -> Dict:
    """Return VADER compound score and label.

    Returns:
        Dict with keys: compound, pos, neu, neg, label.
    """
    if not VADER_AVAILABLE:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral"}
    scores = _VADER.polarity_scores(str(text))
    c = scores["compound"]
    label = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    scores["label"] = label
    return scores


# ---------------------------------------------------------------------------
# 4. Batch Processing
# ---------------------------------------------------------------------------
def analyse_dataframe(df: pd.DataFrame,
                      text_col: str,
                      method: str = "textblob",
                      clean: bool = True) -> pd.DataFrame:
    """Add sentiment columns to a DataFrame.

    Args:
        df      : Input DataFrame.
        text_col: Column containing review/comment text.
        method  : 'textblob' or 'vader'.
        clean   : If True, clean text before scoring.

    Returns:
        Original DataFrame with sentiment columns appended.
    """
    df = df.copy()
    texts = df[text_col].fillna("").astype(str)
    if clean:
        texts = texts.apply(clean_text)

    if method == "vader":
        scores = texts.apply(vader_sentiment)
        df["vader_compound"]  = scores.apply(lambda x: x["compound"])
        df["vader_label"]     = scores.apply(lambda x: x["label"])
    else:
        scores = texts.apply(textblob_sentiment)
        df["tb_polarity"]     = scores.apply(lambda x: x["polarity"])
        df["tb_subjectivity"] = scores.apply(lambda x: x["subjectivity"])
        df["tb_label"]        = scores.apply(lambda x: x["label"])

    print(f"[analyse_dataframe] Scored {len(df):,} rows using {method}.")
    return df


# ---------------------------------------------------------------------------
# 5. Aggregated Sentiment by Group
# ---------------------------------------------------------------------------
def sentiment_by_group(df: pd.DataFrame,
                       group_col: str,
                       score_col: str = "tb_polarity") -> pd.DataFrame:
    """Aggregate mean sentiment score by a grouping column.

    Args:
        df       : DataFrame with sentiment columns already added.
        group_col: Column to group by (e.g., 'category', 'month').
        score_col: Numeric sentiment score column to aggregate.

    Returns:
        Aggregated DataFrame sorted by mean score descending.
    """
    agg = (df.groupby(group_col)[score_col]
             .agg(["mean", "std", "count"])
             .rename(columns={"mean": "avg_sentiment",
                               "std": "std_sentiment",
                               "count": "n_reviews"})
             .sort_values("avg_sentiment", ascending=False)
             .round(4))
    print(f"[sentiment_by_group] Grouped by '{group_col}':\n{agg.to_string()}")
    return agg


# ---------------------------------------------------------------------------
# 6. Convenience wrapper
# ---------------------------------------------------------------------------
def get_sentiment(text: str) -> Dict:
    """Single-text convenience wrapper (TextBlob)."""
    return textblob_sentiment(clean_text(text))


def batch_sentiment(texts: List[str]) -> List[Dict]:
    """Score a list of texts with TextBlob."""
    return [get_sentiment(t) for t in texts]
