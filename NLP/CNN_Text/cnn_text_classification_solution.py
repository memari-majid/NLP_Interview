import numpy as np
from typing import Dict, List

def text_to_sequence(text: str, vocab: Dict[str, int], max_len: int = 10) -> List[int]:
    """Convert text to padded integer sequence."""
    words = text.lower().split()
    sequence = [vocab.get(word, 0) for word in words]  # 0 for unknown words
    
    # Pad or truncate to max_len
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    
    return sequence

def embedding_lookup(sequence: List[int], embedding_matrix: np.ndarray) -> np.ndarray:
    """Look up embeddings for sequence."""
    return np.array([embedding_matrix[idx] for idx in sequence])

def conv1d(embeddings: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 1D convolution with kernel size 3."""
    seq_len, embed_dim = embeddings.shape
    kernel_size = len(kernel)
    
    conv_output = []
    
    # Slide kernel over sequence
    for i in range(seq_len - kernel_size + 1):
        window = embeddings[i:i + kernel_size]  # Shape: (3, embed_dim)
        
        # Element-wise multiply and sum
        conv_value = np.sum(window * kernel[:, np.newaxis])
        conv_output.append(max(0, conv_value))  # ReLU activation
    
    return np.array(conv_output)

def max_pool(conv_output: np.ndarray) -> float:
    """Global max pooling."""
    return np.max(conv_output) if len(conv_output) > 0 else 0.0

def sigmoid(x: float) -> float:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def text_cnn_predict(text: str, vocab: Dict[str, int], 
                    weights: Dict, max_len: int = 10) -> float:
    """CNN forward pass for text classification."""
    
    # 1. Text to sequence
    sequence = text_to_sequence(text, vocab, max_len)
    
    # 2. Embedding lookup
    embeddings = embedding_lookup(sequence, weights['embedding'])
    
    # 3. Convolution
    conv_output = conv1d(embeddings, weights['conv'])
    
    # 4. Max pooling
    pooled = max_pool(conv_output)
    
    # 5. Dense layer + sigmoid
    dense_output = pooled * weights['dense'] + weights['bias']
    probability = sigmoid(dense_output)
    
    return probability

# Example usage and test
def create_sample_weights(vocab_size: int = 100, embed_dim: int = 50):
    """Create sample weights for demonstration."""
    return {
        'embedding': np.random.randn(vocab_size, embed_dim) * 0.1,
        'conv': np.random.randn(3) * 0.1,  # Kernel size 3
        'dense': np.random.randn() * 0.1,
        'bias': 0.0
    }

def test_cnn():
    # Sample vocabulary
    vocab = {
        'good': 1, 'bad': 2, 'movie': 3, 'film': 4, 
        'great': 5, 'terrible': 6, 'love': 7, 'hate': 8
    }
    
    # Sample weights
    weights = create_sample_weights(vocab_size=len(vocab) + 1, embed_dim=10)
    
    # Test sentences
    test_texts = [
        "good movie",
        "bad film", 
        "great love",
        "terrible hate"
    ]
    
    print("CNN Text Classification Results:")
    for text in test_texts:
        prob = text_cnn_predict(text, vocab, weights)
        prediction = "positive" if prob > 0.5 else "negative"
        print(f"'{text}' -> {prob:.3f} ({prediction})")

if __name__ == "__main__":
    test_cnn()
    
    # Demonstrate step-by-step
    print("\nStep-by-step example:")
    vocab = {'good': 1, 'movie': 2}
    text = "good movie"
    
    # Step 1: Tokenization
    sequence = text_to_sequence(text, vocab, max_len=5)
    print(f"Sequence: {sequence}")
    
    # Step 2: Show shapes
    weights = create_sample_weights(vocab_size=10, embed_dim=4)
    embeddings = embedding_lookup(sequence, weights['embedding'])
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Step 3: Convolution
    conv_out = conv1d(embeddings, weights['conv'])
    print(f"Conv output length: {len(conv_out)}")
    
    # Step 4: Max pool
    pooled = max_pool(conv_out)
    print(f"Max pooled value: {pooled:.3f}")