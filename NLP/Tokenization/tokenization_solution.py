import re
from typing import List


TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]", flags=re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words, preserving punctuation as separate tokens.

    Examples:
        >>> tokenize_text("Isn't this great?")
        ["Isn't", 'this', 'great', '?']
    """
    if text is None:
        return []
    return TOKEN_PATTERN.findall(text)


def sentence_tokenize(text: str) -> List[str]:
    """Very simple sentence splitter as a follow-up extension.
    For robust usage, use nltk.sent_tokenize or spaCy.
    """
    if not text:
        return []
    # Split on sentence enders while keeping them
    parts = re.split(r"([.!?])", text)
    # Recombine sentence enders with preceding text
    sentences = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            candidate = (parts[i] + parts[i + 1]).strip()
        else:
            candidate = parts[i].strip()
        if candidate:
            sentences.append(candidate)
    return sentences


if __name__ == "__main__":
    sample = "Natural Language Processing is fascinating! Isn't it?"
    print("Input:", sample)
    print("Tokens:", tokenize_text(sample))
    print("Sentences:", sentence_tokenize(sample))


