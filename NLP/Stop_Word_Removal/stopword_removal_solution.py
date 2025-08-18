from typing import Iterable, List, Optional, Set

try:
    # Prefer NLTK for a standard list
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    ENGLISH_STOPWORDS: Set[str] = set(stopwords.words('english'))
except Exception:  # Fallback minimal list
    ENGLISH_STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'to', 'of',
        'in', 'on', 'for', 'from', 'by', 'is', 'are', 'was', 'were', 'be', 'been',
        'it', 'this', 'that', 'these', 'those', 'as', 'at', 'so', 'than', 'too'
    }


def remove_stopwords(tokens: Iterable[str], extra_stopwords: Optional[Set[str]] = None) -> List[str]:
    """Remove stopwords, preserving order.

    Case-insensitive membership check, preserves original casing in the output.
    """
    stop_set = set(ENGLISH_STOPWORDS)
    if extra_stopwords:
        stop_set |= {w.lower() for w in extra_stopwords}

    cleaned: List[str] = []
    for token in tokens:
        if token and token.lower() not in stop_set:
            cleaned.append(token)
    return cleaned


if __name__ == "__main__":
    example = ["This", "is", "a", "quick", "brown", "fox"]
    print("Input:", example)
    print("Output:", remove_stopwords(example))


