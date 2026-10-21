"""
Sentiment Analysis Utilities
Author: TejeswaraReddy-Dareddy
Date  : 2026-10-21
"""
from textblob import TextBlob

def get_sentiment(text: str) -> dict:
    blob = TextBlob(str(text))
    return {
        'polarity':     blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'label':        'positive' if blob.sentiment.polarity > 0 else ('negative' if blob.sentiment.polarity < 0 else 'neutral')
    }

def batch_sentiment(texts: list) -> list:
    return [get_sentiment(t) for t in texts]
