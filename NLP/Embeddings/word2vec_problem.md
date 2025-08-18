# Problem: Word2Vec Training and Analogy Solver

Implement Word2Vec training and use it for analogy tasks:
1. `train_word2vec(sentences: List[List[str]], embedding_dim: int = 100) -> Word2VecModel`
2. `solve_analogy(model, word_a: str, word_b: str, word_c: str) -> str`
3. `find_similar_words(model, word: str, top_k: int = 5) -> List[Tuple[str, float]]`

Example:
Analogy: "king" - "man" + "woman" = "queen"
Similar to "python": [("java", 0.82), ("programming", 0.78), ("code", 0.75)]

Requirements:
- Implement both CBOW and Skip-gram architectures
- Handle out-of-vocabulary words
- Visualize word embeddings using t-SNE/PCA
- Compare with GloVe and fastText

Follow-ups:
- Subword tokenization for OOV handling
- Contextual embeddings comparison
- Domain-specific word embeddings
