import numpy as np
import math

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function used in GPT."""
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Apply layer normalization."""
    # Calculate mean and variance along last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Scale and shift
    return gamma * normalized + beta

def causal_self_attention(x: np.ndarray, W_qkv: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """Causal self-attention for GPT."""
    seq_len, d_model = x.shape
    
    # Project to Q, K, V
    qkv = x @ W_qkv  # (seq_len, 3 * d_model)
    q, k, v = np.split(qkv, 3, axis=-1)
    
    # Attention scores
    scores = q @ k.T / math.sqrt(d_model)
    
    # Apply causal mask (can't attend to future tokens)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    scores = scores + mask
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Apply attention to values
    attended = attention_weights @ v
    
    # Output projection
    return attended @ W_out

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, 
                W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Feed-forward network with GELU activation."""
    # First layer
    hidden = gelu(x @ W1 + b1)
    
    # Second layer  
    output = hidden @ W2 + b2
    
    return output

def gpt_block(x: np.ndarray, weights: Dict) -> np.ndarray:
    """Single GPT transformer block."""
    seq_len, d_model = x.shape
    
    # 1. Layer norm + self-attention + residual
    norm1 = layer_norm(x, weights['ln1_gamma'], weights['ln1_beta'])
    attn_out = causal_self_attention(norm1, weights['W_qkv'], weights['W_out'])
    x = x + attn_out  # Residual connection
    
    # 2. Layer norm + feed-forward + residual  
    norm2 = layer_norm(x, weights['ln2_gamma'], weights['ln2_beta'])
    ffn_out = feed_forward(norm2, weights['W1'], weights['b1'], weights['W2'], weights['b2'])
    x = x + ffn_out  # Residual connection
    
    return x

def create_gpt_weights(d_model: int = 64, d_ff: int = 256) -> Dict:
    """Create sample weights for GPT block."""
    np.random.seed(42)
    
    return {
        # Layer norm parameters
        'ln1_gamma': np.ones(d_model),
        'ln1_beta': np.zeros(d_model),
        'ln2_gamma': np.ones(d_model), 
        'ln2_beta': np.zeros(d_model),
        
        # Attention weights
        'W_qkv': np.random.randn(d_model, 3 * d_model) * 0.02,
        'W_out': np.random.randn(d_model, d_model) * 0.02,
        
        # Feed-forward weights  
        'W1': np.random.randn(d_model, d_ff) * 0.02,
        'b1': np.zeros(d_ff),
        'W2': np.random.randn(d_ff, d_model) * 0.02,
        'b2': np.zeros(d_model)
    }

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Create sinusoidal positional encodings."""
    pos_enc = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Sine for even indices
            pos_enc[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            
            # Cosine for odd indices
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    
    return pos_enc

def simple_gpt_forward(token_ids: List[int], vocab_size: int = 1000, 
                      d_model: int = 64, num_layers: int = 2) -> np.ndarray:
    """Forward pass through a simple GPT model."""
    seq_len = len(token_ids)
    
    # Token embeddings (random for demo)
    np.random.seed(42)
    embedding_matrix = np.random.randn(vocab_size, d_model) * 0.02
    
    # Get embeddings for input tokens
    x = np.array([embedding_matrix[token_id] for token_id in token_ids])
    
    # Add positional encodings
    pos_enc = positional_encoding(seq_len, d_model)
    x = x + pos_enc
    
    # Pass through transformer blocks
    for layer in range(num_layers):
        weights = create_gpt_weights(d_model)
        x = gpt_block(x, weights)
    
    return x

# Test
if __name__ == "__main__":
    print("GPT Transformer Block Implementation")
    print("=" * 45)
    
    # Test individual components
    seq_len, d_model = 4, 8
    x = np.random.randn(seq_len, d_model)
    
    print("Input shape:", x.shape)
    
    # Test layer norm
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    normed = layer_norm(x, gamma, beta)
    print("After layer norm - mean:", np.mean(normed, axis=-1).round(3))
    print("After layer norm - std:", np.std(normed, axis=-1).round(3))
    
    # Test GELU
    test_vals = np.array([-2, -1, 0, 1, 2])
    gelu_vals = gelu(test_vals)
    print(f"\nGELU test: {test_vals} -> {gelu_vals.round(3)}")
    
    # Test full GPT block
    weights = create_gpt_weights(d_model)
    output = gpt_block(x, weights)
    
    print(f"\nGPT block input shape: {x.shape}")
    print(f"GPT block output shape: {output.shape}")
    print("✓ Shapes preserved through block")
    
    # Test positional encoding
    pos_enc = positional_encoding(seq_len=5, d_model=4)
    print(f"\nPositional encoding shape: {pos_enc.shape}")
    print("First position encoding:", pos_enc[0].round(3))
    
    # Test simple GPT forward pass
    token_ids = [1, 15, 23, 8, 42]  # Sample token sequence
    gpt_output = simple_gpt_forward(token_ids, vocab_size=100, d_model=32, num_layers=2)
    
    print(f"\nSimple GPT test:")
    print(f"Input tokens: {token_ids}")
    print(f"Output shape: {gpt_output.shape}")
    print("✓ Complete GPT forward pass successful")
    
    print("\n" + "=" * 45)
    print("Key GPT Concepts Demonstrated:")
    print("• Causal self-attention (can't see future)")
    print("• Layer normalization for training stability")
    print("• Residual connections to help gradients flow")
    print("• GELU activation in feed-forward networks")
    print("• Positional encodings for sequence position")
