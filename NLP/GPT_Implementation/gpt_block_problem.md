# Problem: GPT Transformer Block

**Time: 30 minutes**

Implement a single GPT transformer block with the standard architecture.

```python
def gpt_block(x: np.ndarray, weights: Dict) -> np.ndarray:
    """
    Single GPT transformer block forward pass.
    
    Architecture:
        x -> LayerNorm -> SelfAttention -> Residual -> LayerNorm -> FFN -> Residual
    
    Args:
        x: Input embeddings (seq_len, d_model)
        weights: Contains attention and FFN weights
        
    Returns:
        Output embeddings (seq_len, d_model)
    """
    pass

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Apply layer normalization.
    norm = (x - mean) / sqrt(variance + eps)
    output = gamma * norm + beta
    """
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, 
                W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Two-layer feed-forward network with GELU activation.
    FFN(x) = GELU(xW1 + b1)W2 + b2
    """
    pass
```

**Requirements:**
- Implement layer normalization with learnable parameters
- Use GELU activation function in FFN
- Add residual connections around attention and FFN
- Apply causal masking in self-attention

**Follow-up:** How would you stack multiple blocks to create full GPT model?
