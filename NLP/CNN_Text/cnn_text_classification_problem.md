# Problem: Text CNN for Classification

**Time: 25 minutes**

Implement a simple CNN for text classification using basic operations.

```python
def text_cnn_predict(text: str, vocab: Dict[str, int], 
                    weights: Dict, max_len: int = 10) -> float:
    """
    Implement forward pass of a text CNN.
    
    Architecture: Embedding -> Conv1D -> MaxPool -> Dense -> Sigmoid
    
    Args:
        text: Input text
        vocab: Word to index mapping  
        weights: {'embedding': [...], 'conv': [...], 'dense': [...]}
        max_len: Maximum sequence length
        
    Returns:
        Probability score (0-1)
    """
    pass
```

**Requirements:**
- Convert text to padded integer sequence
- Implement 1D convolution manually (kernel size 3)
- Apply max pooling across sequence
- Dense layer + sigmoid activation

**Simplifications:** Use numpy only, single filter, binary classification

**Follow-up:** How would you handle variable length sequences efficiently?