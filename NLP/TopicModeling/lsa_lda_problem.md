# Problem: Topic Modeling with LSA and LDA

Implement topic modeling algorithms:
1. `perform_lsa(documents: List[str], num_topics: int = 5) -> LSAModel`
2. `perform_lda(documents: List[str], num_topics: int = 5) -> LDAModel`
3. `extract_topics(model: Union[LSAModel, LDAModel], num_words: int = 10) -> List[List[Tuple[str, float]]]`
4. `get_document_topics(model, document: str) -> List[Tuple[int, float]]`

Example:
Documents: ["Machine learning is fascinating", "Deep learning uses neural networks", ...]
Topics: [(0, [("learning", 0.3), ("neural", 0.2), ...]), (1, [("data", 0.25), ...])]

Requirements:
- LSA using SVD decomposition
- LDA with Gibbs sampling or variational inference  
- Document-topic and topic-word distributions
- Coherence score evaluation

Follow-ups:
- Compare LSA vs LDA vs NMF
- Dynamic topic modeling
- Hierarchical topic models
