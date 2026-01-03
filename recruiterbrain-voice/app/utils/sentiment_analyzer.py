"""
Sentiment analysis for interview answers
"""

from textblob import TextBlob
from typing import Dict


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text
    
    Returns:
        {
            "score": float (-1.0 to 1.0),
            "polarity": float (-1.0 to 1.0),
            "subjectivity": float (0.0 to 1.0)
        }
    """
    blob = TextBlob(text)
    
    return {
        "score": blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
    }


def get_sentiment_label(score: float) -> str:
    """Convert sentiment score to label"""
    if score >= 0.5:
        return "very_positive"
    elif score >= 0.1:
        return "positive"
    elif score >= -0.1:
        return "neutral"
    elif score >= -0.5:
        return "negative"
    else:
        return "very_negative"