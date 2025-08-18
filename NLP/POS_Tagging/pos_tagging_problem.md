# Problem: Part-of-Speech Tagging with Accuracy Metrics

Implement `pos_tag_text(text: str) -> List[Tuple[str, str]]` that:
1. Tags each word with its part-of-speech
2. Handles ambiguous words correctly
3. Returns (word, tag) tuples

Example:
Input: "The quick brown fox jumps over the lazy dog"
Output: [("The", "DT"), ("quick", "JJ"), ("brown", "JJ"), ("fox", "NN"), ...]

Requirements:
- Use Penn Treebank tagset
- Handle sentences with punctuation
- Implement a function to get most common POS for ambiguous words

Follow-ups:
- Extract all nouns/verbs from text
- Find noun phrases using POS patterns
- Compare accuracy of different taggers
