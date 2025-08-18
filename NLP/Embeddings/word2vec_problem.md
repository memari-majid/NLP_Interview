# Problem: Word2Vec Skip-gram

**Time: 20 minutes**

Implement the core Skip-gram training step for Word2Vec.

```python
def skipgram_step(center_word: str, context_word: str, 
                  embeddings: Dict[str, List[float]], 
                  learning_rate: float = 0.01) -> None:
    """
    Perform one training step of Skip-gram Word2Vec.
    
    Update embeddings to make center_word and context_word more similar.
    
    Args:
        center_word: "king" 
        context_word: "queen"
        embeddings: {"king": [0.1, 0.2], "queen": [0.3, 0.4], ...}
        learning_rate: Step size for updates
    """
    pass

def word_similarity(word1: str, word2: str, 
                   embeddings: Dict[str, List[float]]) -> float:
    """
    Calculate cosine similarity between two word embeddings.
    Returns value between -1 and 1.
    """
    pass
```

**Requirements:**
- Implement sigmoid function
- Calculate gradients using dot product  
- Update embeddings using gradient descent
- Handle missing words gracefully

**Follow-up:** How would you implement negative sampling to make this efficient?