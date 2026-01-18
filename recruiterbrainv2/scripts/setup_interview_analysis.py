"""
One-shot setup: download NLTK corpora used by interview analysis.

Run:
  python scripts/setup_interview_analysis.py
"""

import nltk


def main() -> None:
    nltk.download("punkt")
    nltk.download("stopwords")


if __name__ == "__main__":
    main()
