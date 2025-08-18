import math
from typing import Dict, List

def sigmoid(x: float) -> float:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
    
    Used in Word2Vec to convert dot products to probabilities.
    Clamp input to prevent numerical overflow/underflow.
    """
    # Clamp x to prevent overflow in exp() function
    # This is crucial for numerical stability
    clamped_x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-clamped_x))

def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate dot product between two vectors.
    
    Dot product measures similarity between vectors:
    - High positive value = vectors point in same direction (similar)
    - Zero = vectors are orthogonal (unrelated)
    - Negative = vectors point in opposite directions (dissimilar)
    """
    # Element-wise multiplication, then sum
    return sum(a * b for a, b in zip(vec1, vec2))

def skipgram_step(center_word: str, context_word: str, 
                  embeddings: Dict[str, List[float]], 
                  learning_rate: float = 0.01) -> None:
    """
    Perform one training step of Skip-gram Word2Vec.
    
    SKIP-GRAM OBJECTIVE: Given center word, predict context words
    - Makes words that appear in similar contexts have similar embeddings
    - "king" and "queen" both appear near "royal", "crown" -> similar embeddings
    
    TRAINING STEP:
    1. Calculate current similarity between center and context word
    2. If they co-occur, increase their similarity (gradient ascent)
    3. Update both embeddings to be more similar
    
    Args:
        center_word: The word we're using to predict context
        context_word: The word in the context window
        embeddings: Current word embeddings (will be modified in-place)
        learning_rate: How big steps to take during learning
    """
    
    # STEP 1: Handle missing words
    # In real implementation, you'd skip unknown words or use subword tokens
    if center_word not in embeddings or context_word not in embeddings:
        return  # Skip this training example
    
    # STEP 2: Get current embeddings
    center_vec = embeddings[center_word]     # Current embedding for center word
    context_vec = embeddings[context_word]   # Current embedding for context word
    
    # STEP 3: FORWARD PASS - Calculate current similarity
    # Dot product measures how similar the embeddings currently are
    dot_prod = dot_product(center_vec, context_vec)
    
    # Convert to probability using sigmoid
    # High dot product -> high probability of co-occurrence
    prob = sigmoid(dot_prod)
    
    # STEP 4: BACKWARD PASS - Calculate gradients
    # We want to MAXIMIZE the probability of this positive pair
    # Gradient of log(sigmoid(x)) is (1 - sigmoid(x))
    gradient_scale = (1 - prob) * learning_rate
    
    # STEP 5: UPDATE EMBEDDINGS
    # Move embeddings to increase their similarity
    
    # Update center word embedding:
    # Add context_word's embedding scaled by gradient
    for i in range(len(center_vec)):
        embeddings[center_word][i] += gradient_scale * context_vec[i]
    
    # Update context word embedding:
    # Add center_word's embedding scaled by gradient  
    for i in range(len(context_vec)):
        embeddings[context_word][i] += gradient_scale * center_vec[i]
    
    # RESULT: Both embeddings become slightly more similar

def word_similarity(word1: str, word2: str, 
                   embeddings: Dict[str, List[float]]) -> float:
    """
    Calculate cosine similarity between two word embeddings.
    
    Cosine similarity = dot_product(v1, v2) / (||v1|| × ||v2||)
    - Returns value between -1 and 1
    - 1.0 = identical direction (very similar words)
    - 0.0 = orthogonal (unrelated words)
    - -1.0 = opposite direction (opposite meaning)
    """
    
    # STEP 1: Handle missing words
    if word1 not in embeddings or word2 not in embeddings:
        return 0.0  # No similarity if words not in vocabulary
    
    # STEP 2: Get word vectors
    vec1 = embeddings[word1]
    vec2 = embeddings[word2]
    
    # STEP 3: Calculate dot product (numerator)
    dot_prod = dot_product(vec1, vec2)
    
    # STEP 4: Calculate vector norms (denominators)
    # ||v|| = sqrt(sum of squared elements)
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    
    # STEP 5: Handle zero vectors (shouldn't happen with proper training)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # STEP 6: Return cosine similarity
    return dot_prod / (norm1 * norm2)

def find_most_similar(target_word: str, embeddings: Dict[str, List[float]], 
                     top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Find k most similar words to target word.
    
    This is how you'd implement word analogy and similarity queries.
    Core functionality of Word2Vec for exploration and evaluation.
    """
    if target_word not in embeddings:
        return []  # Target word not in vocabulary
    
    # Calculate similarity with all other words
    similarities = []
    
    for word in embeddings:
        if word != target_word:  # Don't compare word with itself
            sim = word_similarity(target_word, word, embeddings)
            similarities.append((word, sim))
    
    # Sort by similarity (highest first) and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# INTERVIEW DEMONSTRATION
if __name__ == "__main__":
    print("WORD2VEC SKIP-GRAM - Interview Demo")
    print("=" * 45)
    
    # STEP 1: Initialize sample embeddings
    # In practice, these start random and get trained on large corpus
    print("1. INITIAL EMBEDDINGS")
    print("-" * 20)
    
    embeddings = {
        'king': [0.1, 0.2, 0.3],      # Royal, male concept
        'queen': [0.15, 0.25, 0.35],  # Royal, female concept  
        'man': [0.2, 0.1, 0.4],       # Male concept
        'woman': [0.25, 0.15, 0.45],  # Female concept
        'royal': [0.12, 0.22, 0.32]   # Royal concept
    }
    
    print("Initial word embeddings:")
    for word, vec in embeddings.items():
        print(f"  {word}: {vec}")
    
    # STEP 2: Show initial similarities
    print(f"\n2. INITIAL SIMILARITIES")
    print("-" * 25)
    
    word_pairs = [('king', 'queen'), ('king', 'man'), ('man', 'woman')]
    print("Initial similarities:")
    for w1, w2 in word_pairs:
        sim = word_similarity(w1, w2, embeddings)
        print(f"  {w1}-{w2}: {sim:.3f}")
    
    # STEP 3: Simulate training with co-occurrence data
    print(f"\n3. TRAINING SIMULATION")
    print("-" * 25)
    
    # These pairs co-occur in training corpus
    training_pairs = [
        ('king', 'royal'), ('queen', 'royal'),    # Royal context
        ('man', 'king'), ('woman', 'queen'),      # Gender-role context
        ('king', 'queen'),                        # Royal pair
        ('man', 'woman')                          # Gender pair
    ]
    
    print("Training on co-occurrence pairs:")
    for center, context in training_pairs:
        print(f"  {center} <-> {context}")
    
    # Perform training steps
    print(f"\nPerforming {len(training_pairs) * 10} training steps...")
    
    for epoch in range(10):  # Multiple epochs
        for center_word, context_word in training_pairs:
            # Train both directions (Skip-gram)
            skipgram_step(center_word, context_word, embeddings, learning_rate=0.1)
            skipgram_step(context_word, center_word, embeddings, learning_rate=0.1)
    
    # STEP 4: Show results after training
    print(f"\n4. RESULTS AFTER TRAINING")
    print("-" * 30)
    
    print("Final similarities:")
    for w1, w2 in word_pairs:
        sim = word_similarity(w1, w2, embeddings)
        print(f"  {w1}-{w2}: {sim:.3f}")
    
    # STEP 5: Demonstrate similarity search
    print(f"\n5. SIMILARITY SEARCH")
    print("-" * 25)
    
    for word in ['king', 'man']:
        similar_words = find_most_similar(word, embeddings, top_k=2)
        print(f"\nMost similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.3f}")
    
    print(f"\n" + "=" * 45)
    print("INTERVIEW EXPLANATION POINTS:")
    print("=" * 45)
    print("• Skip-gram predicts context from center word")
    print("• Words in similar contexts get similar embeddings")
    print("• Dot product measures embedding similarity")
    print("• Gradient descent makes co-occurring words more similar")
    print("• Cosine similarity normalizes for vector magnitude")
    print("• Training requires negative sampling in practice")
    
    print(f"\n" + "=" * 45)
    print("COMMON FOLLOW-UP QUESTIONS:")
    print("=" * 45)
    print("Q: How do you handle rare words?")
    print("A: Subword tokenization (BPE) or character-level embeddings")
    print()
    print("Q: Skip-gram vs CBOW?")
    print("A: Skip-gram: center->context, CBOW: context->center")
    print()
    print("Q: How to train efficiently?")
    print("A: Negative sampling, hierarchical softmax")
    print()
    print("Q: Static vs contextual embeddings?")
    print("A: Word2Vec static, BERT contextual (same word, different meanings)")