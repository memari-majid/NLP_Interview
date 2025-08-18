# Problem: Bag of Words from Scratch

Implement bag-of-words representation from scratch:
1. `BagOfWords(max_features: int, binary: bool)` - BoW vectorizer class
2. `fit(documents: List[str])` - Build vocabulary from documents
3. `transform(documents: List[str]) -> np.ndarray` - Convert to BoW vectors
4. `get_feature_names() -> List[str]` - Get vocabulary words

Example:
Documents: ["I love NLP", "NLP is great", "I love programming"]
Vocabulary: ["I", "love", "NLP", "is", "great", "programming"]
BoW matrix: [[1,1,1,0,0,0], [0,0,1,1,1,0], [1,1,0,0,0,1]]

Requirements:
- Handle different tokenization strategies
- Support binary vs count-based representation  
- Implement vocabulary pruning (min_df, max_df)
- Compare with character-level BoW
- Handle out-of-vocabulary words

Follow-ups:
- Add n-gram support (bigrams, trigrams)
- Implement feature selection (chi-square, mutual information)
- Memory-efficient sparse matrix representation
- Streaming/incremental vocabulary building
