import math
from typing import List, Tuple, Set
from collections import Counter

def create_bow_vector(documents: List[str]) -> Tuple[List[str], List[List[int]]]:
    """Create bag-of-words representation."""
    if not documents:
        return [], []
    
    # Build vocabulary
    vocab_set = set()
    for doc in documents:
        words = doc.lower().split()
        vocab_set.update(words)
    
    vocabulary = sorted(list(vocab_set))  # Sort for consistency
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    # Create vectors
    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_counts = Counter(words)
        
        # Create vector with counts
        vector = [word_counts.get(word, 0) for word in vocabulary]
        vectors.append(vector)
    
    return vocabulary, vectors

def cosine_similarity(vec1: List[int], vec2: List[int]) -> float:
    """Calculate cosine similarity between two BoW vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Norms
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def find_most_similar(documents: List[str], query_idx: int) -> int:
    """Find document most similar to query document."""
    vocab, vectors = create_bow_vector(documents)
    
    if query_idx >= len(vectors):
        return -1
    
    query_vector = vectors[query_idx]
    max_similarity = -1
    most_similar_idx = -1
    
    for i, doc_vector in enumerate(vectors):
        if i != query_idx:  # Don't compare with itself
            similarity = cosine_similarity(query_vector, doc_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_idx = i
    
    return most_similar_idx

# Test
if __name__ == "__main__":
    documents = [
        "I love natural language processing",
        "Machine learning is fascinating", 
        "I love machine learning",
        "Natural language processing is great"
    ]
    
    vocab, vectors = create_bow_vector(documents)
    
    print("Vocabulary:", vocab[:10])  # Show first 10 words
    print("\nDocument vectors:")
    for i, (doc, vec) in enumerate(zip(documents, vectors)):
        non_zero = [(vocab[j], count) for j, count in enumerate(vec) if count > 0]
        print(f"Doc {i}: {non_zero}")
    
    # Test similarity
    sim = cosine_similarity(vectors[0], vectors[2])
    print(f"\nSimilarity between doc 0 and doc 2: {sim:.3f}")
    
    # Find most similar
    most_similar = find_most_similar(documents, 0)
    print(f"Most similar to doc 0: doc {most_similar}")
    print(f"'{documents[most_similar]}'")
