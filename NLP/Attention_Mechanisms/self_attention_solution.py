import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation.
    
    Why stable? Subtracting max prevents overflow when exponentiating large numbers.
    This is critical for attention weights which can have large values.
    """
    # Subtract maximum value for numerical stability
    # This doesn't change the relative probabilities but prevents exp() overflow
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # Normalize to get probabilities (sum to 1 along specified axis)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X: np.ndarray, d_k: int) -> np.ndarray:
    """
    Implement scaled dot-product self-attention mechanism.
    
    This is the CORE of all transformer models (BERT, GPT, etc.)
    
    Formula: Attention(Q,K,V) = softmax(QK^T / √d_k)V
    
    Args:
        X: Input embeddings (seq_len, d_model)
        d_k: Dimension of queries and keys (for scaling)
    
    Returns:
        Attention output (seq_len, d_model)
    """
    seq_len, d_model = X.shape
    
    # STEP 1: Initialize weight matrices
    # In practice, these are learned parameters
    # Using small random values for demonstration
    np.random.seed(42)  # For reproducible results in interview
    W_q = np.random.randn(d_model, d_k) * 0.1    # Query projection
    W_k = np.random.randn(d_model, d_k) * 0.1    # Key projection  
    W_v = np.random.randn(d_model, d_model) * 0.1 # Value projection
    
    # STEP 2: Create Query, Key, Value matrices
    # These are linear transformations of the input
    # Q: what we're looking for, K: what we're comparing against, V: what we return
    Q = X @ W_q  # Shape: (seq_len, d_k)
    K = X @ W_k  # Shape: (seq_len, d_k)  
    V = X @ W_v  # Shape: (seq_len, d_model)
    
    # STEP 3: Calculate attention scores
    # QK^T gives us similarity between all pairs of positions
    scores = Q @ K.T  # Shape: (seq_len, seq_len)
    
    # STEP 4: Scale by sqrt(d_k)
    # Prevents dot products from getting too large (which makes softmax too peaked)
    # This scaling is CRUCIAL for stable training
    scores = scores / np.sqrt(d_k)
    
    # STEP 5: Apply softmax to get attention weights
    # Each row sums to 1.0 - these are the attention probabilities
    # Higher score = more attention to that position
    attention_weights = softmax(scores, axis=-1)
    
    # STEP 6: Apply attention to values
    # Weighted sum of value vectors based on attention weights
    # This is where information actually flows between positions
    output = attention_weights @ V  # Shape: (seq_len, d_model)
    
    return output

def self_attention_with_mask(X: np.ndarray, d_k: int, mask: np.ndarray = None) -> np.ndarray:
    """
    Self-attention with causal masking (for GPT-style models).
    
    Causal mask prevents positions from attending to future positions.
    Essential for autoregressive generation (predict next token).
    """
    seq_len, d_model = X.shape
    
    # Same weight initialization as before
    np.random.seed(42)
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    
    # Create Q, K, V matrices
    Q = X @ W_q
    K = X @ W_k  
    V = X @ W_v
    
    # Calculate attention scores
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # STEP: Apply mask BEFORE softmax
    # Masked positions get large negative values (-inf conceptually)
    # After softmax, these become ~0 probability
    if mask is not None:
        # Add large negative value to masked positions
        scores = scores + (mask * -1e9)
    
    # Apply softmax and compute output
    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V
    
    return output

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask for autoregressive attention.
    
    Used in GPT to prevent "looking into the future" during training.
    
    Returns:
        Mask matrix where 1 = mask (don't attend), 0 = allow
    """
    # Create lower triangular matrix (1s below and on diagonal)
    # This allows attending to current and previous positions only
    lower_triangle = np.tril(np.ones((seq_len, seq_len)))
    
    # Convert to mask format: 0 = allow attention, 1 = mask
    return 1 - lower_triangle

def multi_head_attention(X: np.ndarray, d_k: int, num_heads: int = 8) -> np.ndarray:
    """
    Multi-head self-attention (simplified version).
    
    Key insight: Multiple attention heads can focus on different aspects
    - Head 1 might focus on syntax
    - Head 2 might focus on semantics
    - Head 3 might focus on long-range dependencies
    """
    seq_len, d_model = X.shape
    
    # REQUIREMENT: d_model must be divisible by num_heads
    # Each head gets d_model/num_heads dimensions
    assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
    
    head_dim = d_model // num_heads
    outputs = []
    
    # STEP: Apply attention for each head independently
    for head in range(num_heads):
        # Each head operates on a subset of the input dimensions
        start_idx = head * head_dim
        end_idx = start_idx + head_dim
        X_head = X[:, start_idx:end_idx]
        
        # Apply self-attention to this head's slice
        head_output = self_attention(X_head, head_dim)
        outputs.append(head_output)
    
    # STEP: Concatenate all head outputs
    # This gives us the full d_model dimensional output
    return np.concatenate(outputs, axis=-1)

# INTERVIEW DEMONSTRATION CODE
if __name__ == "__main__":
    print("Self-Attention Mechanism - Interview Demo")
    print("=" * 50)
    
    # STEP 1: Create sample input embeddings
    # In practice, these come from token embeddings + positional encodings
    X = np.array([
        [1.0, 0.5, 0.2, 0.1],  # Token 1 embedding
        [0.8, 1.0, 0.3, 0.2],  # Token 2 embedding  
        [0.6, 0.4, 1.0, 0.5],  # Token 3 embedding
    ])
    
    print("INPUT EMBEDDINGS:")
    print(f"Shape: {X.shape} (3 tokens, 4 dimensions each)")
    print("X =")
    print(X)
    
    # STEP 2: Apply self-attention
    print(f"\nAPPLYING SELF-ATTENTION:")
    output = self_attention(X, d_k=4)
    
    print("OUTPUT:")
    print(f"Shape: {output.shape} (same as input)")
    print("Output =")
    print(output.round(3))
    
    # STEP 3: Demonstrate causal masking
    print(f"\n" + "=" * 50)
    print("CAUSAL MASKING (for GPT-style models)")
    print("=" * 50)
    
    # Create and show causal mask
    mask = create_causal_mask(3)
    print("Causal mask (1 = masked, 0 = allowed):")
    print(mask)
    print("\nInterpretation:")
    print("• Position 0 can only attend to itself")
    print("• Position 1 can attend to positions 0 and 1")  
    print("• Position 2 can attend to positions 0, 1, and 2")
    
    # Apply masked attention
    masked_output = self_attention_with_mask(X, d_k=4, mask=mask)
    print(f"\nMasked attention output:")
    print(masked_output.round(3))
    
    # STEP 4: Show attention weight visualization
    print(f"\n" + "=" * 50)
    print("ATTENTION WEIGHTS VISUALIZATION")
    print("=" * 50)
    
    # Manual calculation to show attention weights
    np.random.seed(42)
    W_q = W_k = np.eye(4) * 0.5  # Simple weights for demonstration
    Q = K = X @ W_q
    
    scores = Q @ K.T / np.sqrt(4)
    weights = softmax(scores, axis=-1)
    
    print("Attention weights matrix:")
    print(weights.round(3))
    print("\nHow to read this matrix:")
    print("• Row i = what position i attends to")
    print("• Column j = how much attention position j receives")
    print("• Each row sums to 1.0")
    print("• Diagonal elements = self-attention")
    
    print(f"\n" + "=" * 50)
    print("INTERVIEW TALKING POINTS:")
    print("=" * 50)
    print("✓ Attention allows each position to look at all other positions")
    print("✓ Scaling by √d_k prevents gradients from vanishing")
    print("✓ Softmax ensures attention weights sum to 1")
    print("✓ Causal masking is crucial for autoregressive models")
    print("✓ Multi-head attention captures different types of relationships")
    print("✓ Time complexity: O(n²d) where n=seq_len, d=d_model")
    print("✓ Space complexity: O(n²) for attention matrix")