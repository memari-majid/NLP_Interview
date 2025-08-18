# Problem: TF-IDF Implementation and Document Similarity

Implement a TF-IDF system with:
1. `calculate_tfidf(documents: List[str]) -> np.ndarray` - Returns TF-IDF matrix
2. `find_similar_documents(query: str, documents: List[str], top_k: int = 3) -> List[Tuple[int, float]]` - Returns top-k similar documents

Example:
Documents = ["The cat sat on mat", "The dog sat on log", "Cats and dogs are pets"]
Query = "cat on mat"
Output: [(0, 0.89), (2, 0.52), (1, 0.31)]  # (doc_index, similarity_score)

Requirements:
- Implement from scratch (formulas) and using sklearn
- Handle edge cases (empty docs, single word)
- Compare with BM25 scoring

Follow-ups:
- Add IDF smoothing options
- Implement sublinear TF scaling
- Optimize for large document collections
