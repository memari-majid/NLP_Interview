# Problem: Remove Stopwords (Configurable)

Implement `remove_stopwords(tokens: List[str], extra_stopwords: Optional[Set[str]] = None) -> List[str]` that removes English stopwords from a token list.

Example
Input tokens: ["this", "is", "a", "quick", "brown", "fox"]
Output: ["quick", "brown", "fox"]

Requirements
- Use a standard English stopword list.
- Allow passing custom stopwords via `extra_stopwords`.
- Preserve original token order.

Follow-ups
- Case-insensitive removal while preserving original casing.
- Support multiple languages.


