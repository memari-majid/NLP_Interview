# Problem: TF-IDF Implementation

**Time: 30 minutes**

Implement TF-IDF from scratch to find document similarity.

```python
def compute_tfidf(documents: List[str]) -> List[Dict[str, float]]:
    """
    Compute TF-IDF vectors for documents.
    
    Input: ["cat sat mat", "dog sat log", "cat dog"]
    Output: [{"cat": 0.47, "sat": 0.0, "mat": 0.69}, {...}, {...}]
    
    TF = term_freq / total_terms_in_doc
    IDF = log(total_docs / docs_containing_term)
    TF-IDF = TF * IDF
    """
    pass

def find_similar_documents(documents: List[str], query: str) -> int:
    """
    Return index of most similar document to query using cosine similarity.
    """
    pass
```

**Requirements:**
- Implement TF-IDF calculation from scratch
- Use cosine similarity for document comparison
- Handle empty documents gracefully

**Follow-up:** How would you optimize this for millions of documents?