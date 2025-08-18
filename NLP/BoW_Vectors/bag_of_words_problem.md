# Problem: Bag of Words from Scratch

**Time: 20 minutes**

Implement a basic bag-of-words vectorizer.

```python
def create_bow_vector(documents: List[str]) -> Tuple[List[str], List[List[int]]]:
    """
    Create bag-of-words representation.
    
    Input: ["I love NLP", "NLP is great", "I love programming"]
    Output: 
        vocabulary: ["I", "love", "NLP", "is", "great", "programming"]
        vectors: [[1,1,1,0,0,0], [0,0,1,1,1,0], [1,1,0,0,0,1]]
    
    Returns:
        (vocabulary, document_vectors)
    """
    pass

def cosine_similarity(vec1: List[int], vec2: List[int]) -> float:
    """
    Calculate cosine similarity between two BoW vectors.
    Return value between 0 and 1.
    """
    pass
```

**Requirements:**
- Build vocabulary from all documents
- Count word occurrences in each document
- Handle empty documents
- Implement cosine similarity for vector comparison

**Follow-up:** How would you handle very large vocabularies efficiently?