# Problem: N-gram Language Model

**Time: 25 minutes**

Implement a simple bigram language model with probability calculation.

```python
def build_bigram_model(texts: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Build bigram language model.
    
    Input: ["the cat sat", "the dog sat"]
    Output: {
        "the": {"cat": 0.5, "dog": 0.5},
        "cat": {"sat": 1.0},
        "dog": {"sat": 1.0}
    }
    
    Returns: Dictionary mapping word -> {next_word: probability}
    """
    pass

def generate_text(model: Dict, start_word: str, length: int = 5) -> str:
    """
    Generate text using the bigram model.
    Pick most probable next word at each step.
    """
    pass
```

**Requirements:**
- Count bigram frequencies across all texts
- Calculate conditional probabilities P(w2|w1)
- Handle unseen bigrams (return empty dict)
- Generate coherent text sequences

**Follow-up:** How would you add smoothing for unseen n-grams?