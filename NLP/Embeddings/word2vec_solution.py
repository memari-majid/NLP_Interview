import math
from typing import Dict, List

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """Calculate dot product of two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))

def skipgram_step(center_word: str, context_word: str, 
                  embeddings: Dict[str, List[float]], 
                  learning_rate: float = 0.01) -> None:
    """Perform one Skip-gram training step."""
    
    if center_word not in embeddings or context_word not in embeddings:
        return
    
    center_vec = embeddings[center_word]
    context_vec = embeddings[context_word]
    
    # Forward pass
    dot_prod = dot_product(center_vec, context_vec)
    prob = sigmoid(dot_prod)
    
    # Gradient and update
    gradient_scale = (1 - prob) * learning_rate
    
    for i in range(len(center_vec)):
        embeddings[center_word][i] += gradient_scale * context_vec[i]
        embeddings[context_word][i] += gradient_scale * center_vec[i]

def word_similarity(word1: str, word2: str, 
                   embeddings: Dict[str, List[float]]) -> float:
    """Calculate cosine similarity between two word embeddings."""
    
    if word1 not in embeddings or word2 not in embeddings:
        return 0.0
    
    vec1, vec2 = embeddings[word1], embeddings[word2]
    
    dot_prod = dot_product(vec1, vec2)
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_prod / (norm1 * norm2)

# Test
if __name__ == "__main__":
    embeddings = {
        'king': [0.1, 0.2, 0.3],
        'queen': [0.15, 0.25, 0.35],
        'man': [0.2, 0.1, 0.4],
        'woman': [0.25, 0.15, 0.45]
    }
    
    print("Before training:")
    print(f"king-queen: {word_similarity('king', 'queen', embeddings):.3f}")
    
    # Train
    for _ in range(10):
        skipgram_step('king', 'queen', embeddings, 0.1)
    
    print("After training:")
    print(f"king-queen: {word_similarity('king', 'queen', embeddings):.3f}")