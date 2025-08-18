# Problem: Stemming vs Lemmatization

Implement two functions:
1. `stem_words(words: List[str]) -> List[str]` - Porter stemming
2. `lemmatize_words(words: List[str], pos_tags: Optional[List[str]] = None) -> List[str]` - With POS awareness

Example:
Input: ["running", "ran", "runs", "better", "best"]
Stemmed: ["run", "ran", "run", "better", "best"]
Lemmatized: ["run", "run", "run", "good", "good"]

Requirements:
- Show the difference between stemming and lemmatization
- Handle POS tags for better lemmatization
- Compare outputs side by side

Follow-ups:
- Which method to use for information retrieval vs text classification?
- Performance implications?
