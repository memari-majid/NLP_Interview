# Problem: Self-Attention from Scratch

**Time: 25 minutes**

Implement scaled dot-product self-attention mechanism.

```python
def self_attention(X: np.ndarray, d_k: int) -> np.ndarray:
    """
    Implement self-attention mechanism.
    
    Args:
        X: Input matrix (seq_len, d_model)
        d_k: Key/Query dimension
        
    Returns:
        Attention output (seq_len, d_model)
        
    Steps:
        1. Create Q, K, V matrices from X
        2. Compute attention scores: QK^T / sqrt(d_k)
        3. Apply softmax to get attention weights
        4. Return weighted sum of values: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    pass

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask to prevent attending to future tokens.
    Return lower triangular matrix of ones.
    """
    pass
```

**Requirements:**
- Implement Q, K, V transformations using random weights
- Calculate attention scores with scaling
- Apply softmax row-wise
- Handle causal masking for autoregressive models

**Follow-up:** How would you implement multi-head attention?
