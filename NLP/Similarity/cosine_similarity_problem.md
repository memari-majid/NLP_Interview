# Problem: Text Similarity Metrics

Implement multiple similarity metrics:
1. `cosine_similarity(text1: str, text2: str) -> float`
2. `jaccard_similarity(text1: str, text2: str) -> float`  
3. `semantic_similarity(text1: str, text2: str) -> float` # Using word embeddings

Example:
text1 = "The cat sat on the mat"
text2 = "The feline rested on the rug"
cosine_sim = 0.45 (bag-of-words)
semantic_sim = 0.82 (word2vec/embeddings)

Requirements:
- Compare bag-of-words vs embeddings approaches
- Handle synonyms and semantic relationships
- Implement efficient similarity for large text collections

Follow-ups:
- Add Levenshtein distance for character-level similarity
- Implement MinHash for approximate similarity
- Build a text deduplication system
