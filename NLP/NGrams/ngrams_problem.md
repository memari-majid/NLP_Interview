# Problem: N-Gram Language Model

Implement an n-gram language model with the following functions:
1. `build_ngram_model(texts: List[str], n: int = 3) -> NGramModel`
2. `calculate_perplexity(model: NGramModel, text: str) -> float`
3. `generate_text(model: NGramModel, seed: str, length: int = 50) -> str`
4. `apply_smoothing(model: NGramModel, method: str = 'laplace') -> NGramModel`

Example:
Training: ["The cat sat on the mat", "The dog sat on the log"]
Generate from "The cat": "The cat sat on the mat"
Perplexity: 2.34

Requirements:
- Handle different n values (unigram, bigram, trigram)
- Implement Laplace/Good-Turing smoothing
- Calculate probability distributions
- Handle unseen n-grams gracefully

Follow-ups:
- Kneser-Ney smoothing
- Back-off models
- Character-level n-grams for morphology
