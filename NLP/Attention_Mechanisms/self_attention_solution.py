import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X: np.ndarray, d_k: int) -> np.ndarray:
    """Implement scaled dot-product self-attention."""
    seq_len, d_model = X.shape
    
    # Initialize random weight matrices
    np.random.seed(42)  # For reproducible results
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1  
    W_v = np.random.randn(d_model, d_model) * 0.1
    
    # Create Query, Key, Value matrices
    Q = X @ W_q  # (seq_len, d_k)
    K = X @ W_k  # (seq_len, d_k)
    V = X @ W_v  # (seq_len, d_model)
    
    # Calculate attention scores
    scores = Q @ K.T  # (seq_len, seq_len)
    scores = scores / np.sqrt(d_k)  # Scale by sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Apply attention to values
    output = attention_weights @ V  # (seq_len, d_model)
    
    return output

def self_attention_with_mask(X: np.ndarray, d_k: int, mask: np.ndarray = None) -> np.ndarray:
    """Self-attention with optional causal mask."""
    seq_len, d_model = X.shape
    
    # Same as before until attention scores
    np.random.seed(42)
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    
    Q = X @ W_q
    K = X @ W_k  
    V = X @ W_v
    
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # Apply mask (set masked positions to large negative value)
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V
    
    return output

def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create causal mask for autoregressive attention."""
    # Lower triangular matrix (can attend to current and previous tokens)
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    # Convert to mask format (0 for allowed, 1 for masked)
    return 1 - mask

def multi_head_attention(X: np.ndarray, d_k: int, num_heads: int = 8) -> np.ndarray:
    """Multi-head self-attention (simplified)."""
    seq_len, d_model = X.shape
    
    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    head_dim = d_model // num_heads
    outputs = []
    
    # Apply attention for each head
    for head in range(num_heads):
        # Use subset of features for this head
        start_idx = head * head_dim
        end_idx = start_idx + head_dim
        X_head = X[:, start_idx:end_idx]
        
        # Apply self-attention (using head_dim for d_k)
        head_output = self_attention(X_head, head_dim)
        outputs.append(head_output)
    
    # Concatenate all heads
    return np.concatenate(outputs, axis=-1)

# Test functions
def test_attention():
    # Create sample input (3 tokens, 4 dimensions each)
    X = np.array([
        [1.0, 0.5, 0.2, 0.1],  # Token 1
        [0.8, 1.0, 0.3, 0.2],  # Token 2  
        [0.6, 0.4, 1.0, 0.5],  # Token 3
    ])
    
    print("Input matrix X:")
    print(X)
    print(f"Shape: {X.shape}")
    
    # Apply self-attention
    output = self_attention(X, d_k=4)
    print(f"\nSelf-attention output:")
    print(output)
    print(f"Shape: {output.shape}")
    
    # Test with causal mask
    mask = create_causal_mask(3)
    print(f"\nCausal mask:")
    print(mask)
    
    masked_output = self_attention_with_mask(X, d_k=4, mask=mask)
    print(f"\nMasked self-attention output:")
    print(masked_output)

if __name__ == "__main__":
    test_attention()
    
    # Demonstrate attention weights
    print("\n" + "="*40)
    print("Attention Weight Visualization")
    print("="*40)
    
    # Simple example to show attention
    X = np.array([[1, 0], [0, 1], [1, 1]])  # 3 tokens, 2 dims
    
    # Manual calculation for demonstration
    W_q = W_k = np.eye(2) * 0.5  # Simple weights
    Q = K = X @ W_q
    
    scores = Q @ K.T / np.sqrt(2)
    weights = softmax(scores, axis=-1)
    
    print("Attention weights matrix:")
    print(weights.round(3))
    print("\nInterpretation:")
    print("- Row i shows what token i attends to")
    print("- Each row sums to 1.0")
    print("- Higher values = more attention")
