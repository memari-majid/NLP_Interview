# Problem: Text Tokenization

**Time: 15 minutes**

Implement a function that tokenizes text into words while handling edge cases.

```python
def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    Handle contractions, punctuation, and empty strings.
    
    Examples:
    tokenize("Hello world!") -> ["Hello", "world", "!"]
    tokenize("don't") -> ["don't"]  # Keep contractions intact
    tokenize("") -> []
    """
    pass
```

**Requirements:**
- Split on whitespace and punctuation (except apostrophes in contractions)
- Handle empty/None input
- Preserve contractions like "don't", "I'm"

**Follow-up:** How would you handle different languages or subword tokenization?