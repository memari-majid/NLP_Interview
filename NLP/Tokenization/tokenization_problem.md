# Problem: Word Tokenization (Preserve Punctuation)

Implement a function `tokenize_text(text: str) -> List[str]` that splits input text into word tokens while preserving punctuation as separate tokens.

Example
Input: "Natural Language Processing is fascinating! Isn't it?"
Output: ["Natural", "Language", "Processing", "is", "fascinating", "!", "Isn't", "it", "?"]

Requirements
- Treat contractions as single tokens (e.g., "Isn't").
- Preserve punctuation as separate tokens.
- Unicode friendly.

Follow-ups
- Add sentence tokenization.
- Make tokenizer configurable (keep/remove punctuation).


