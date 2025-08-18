import re
from typing import List

def tokenize(text: str) -> List[str]:
    """Tokenize text into words, preserving contractions."""
    if not text:
        return []
    
    # Pattern: word characters + apostrophes OR single punctuation
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    return re.findall(pattern, text)

def tokenize_simple(text: str) -> List[str]:
    """Alternative: simple split-based approach."""
    if not text:
        return []
    
    # Replace punctuation with spaces, except apostrophes
    cleaned = re.sub(r"[^\w\s']", " ", text)
    return cleaned.split()

# Test cases
def test_tokenization():
    test_cases = [
        ("Hello world!", ["Hello", "world", "!"]),
        ("don't go", ["don't", "go"]),
        ("I'm happy.", ["I'm", "happy", "."]),
        ("", []),
        ("one,two;three", ["one", ",", "two", ";", "three"])
    ]
    
    for text, expected in test_cases:
        result = tokenize(text)
        print(f"Input: '{text}' -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_tokenization()