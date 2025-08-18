import math
from typing import List, Tuple, Set
from collections import Counter

def create_bow_vector(documents: List[str]) -> Tuple[List[str], List[List[int]]]:
    """
    Create bag-of-words representation from scratch.
    
    BAG-OF-WORDS CONCEPT:
    - Represent text as vector of word counts
    - "Order doesn't matter, just word presence/frequency"
    - Foundation of many NLP systems before embeddings
    
    STEPS:
    1. Build vocabulary (all unique words)
    2. For each document, count occurrences of each vocab word
    3. Return vocabulary and count vectors
    
    INTERVIEW INSIGHT: Simple but effective. Foundation for TF-IDF.
    """
    
    # STEP 1: Handle edge case
    if not documents:
        return [], []
    
    # STEP 2: Build vocabulary from all documents
    # We need consistent vocabulary across all documents for vector comparison
    vocab_set = set()
    
    for doc in documents:
        # Simple tokenization: lowercase and split on whitespace
        # In interviews, mention this could be more sophisticated
        words = doc.lower().split()
        vocab_set.update(words)
    
    # STEP 3: Create ordered vocabulary
    # Sorting ensures consistent feature ordering across runs
    vocabulary = sorted(list(vocab_set))
    
    # Create word-to-index mapping for efficient lookup
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    # STEP 4: Convert each document to vector
    vectors = []
    
    for doc in documents:
        # Tokenize document
        words = doc.lower().split()
        
        # Count word occurrences in this document
        word_counts = Counter(words)
        
        # Create vector: count for each vocabulary word
        # Index i contains count of vocabulary[i] in this document
        vector = []
        for word in vocabulary:
            count = word_counts.get(word, 0)  # 0 if word not in document
            vector.append(count)
        
        vectors.append(vector)
    
    return vocabulary, vectors

def cosine_similarity(vec1: List[int], vec2: List[int]) -> float:
    """
    Calculate cosine similarity between two BoW vectors.
    
    COSINE SIMILARITY INTUITION:
    - Measures angle between vectors, not magnitude
    - Good for text: "I love cats" vs "I really really love cats"
    - Both have same direction (similar meaning) despite different lengths
    
    FORMULA: cos(θ) = (A·B) / (||A|| × ||B||)
    
    INTERVIEW TIP: Always explain why cosine > Euclidean for text
    """
    
    # STEP 1: Input validation
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    # STEP 2: Calculate dot product (numerator)
    # Sum of element-wise products
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # STEP 3: Calculate vector norms (denominators)
    # ||v|| = sqrt(sum of squared elements) = vector magnitude
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    
    # STEP 4: Handle zero vectors (edge case)
    # Zero vector has no direction, so similarity is undefined
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # STEP 5: Return cosine similarity
    # Result is between -1 (opposite) and 1 (identical)
    # For BoW with counts, result is between 0 and 1
    return dot_product / (norm1 * norm2)

def find_most_similar(documents: List[str], query_idx: int) -> int:
    """
    Find document most similar to query document.
    
    REAL-WORLD APPLICATION:
    - Document search and retrieval
    - "Find documents similar to this one"
    - Recommendation systems ("users who read this also read...")
    
    ALGORITHM:
    1. Convert all documents to BoW vectors
    2. Calculate similarity between query and each document
    3. Return index of most similar document
    """
    
    # STEP 1: Convert documents to BoW representation
    vocab, vectors = create_bow_vector(documents)
    
    # STEP 2: Validate query index
    if query_idx >= len(vectors) or query_idx < 0:
        return -1  # Invalid query index
    
    # STEP 3: Get query vector
    query_vector = vectors[query_idx]
    
    # STEP 4: Calculate similarity with all other documents
    max_similarity = -1  # Start with impossible low value
    most_similar_idx = -1
    
    for i, doc_vector in enumerate(vectors):
        # Don't compare document with itself
        if i != query_idx:
            similarity = cosine_similarity(query_vector, doc_vector)
            
            # Track the highest similarity found
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_idx = i
    
    return most_similar_idx

def analyze_vocabulary_distribution(documents: List[str]) -> Dict[str, int]:
    """
    Analyze vocabulary characteristics - good for follow-up questions.
    
    INTERVIEW INSIGHTS:
    - Most words appear very rarely (Zipf's law)
    - Top 100 words account for ~50% of text
    - Vocabulary size grows with more documents
    """
    vocab, _ = create_bow_vector(documents)
    
    # Count how often each word appears across documents
    word_doc_counts = {}
    for word in vocab:
        count = sum(1 for doc in documents if word in doc.lower())
        word_doc_counts[word] = count
    
    return word_doc_counts

# COMPREHENSIVE INTERVIEW DEMO
if __name__ == "__main__":
    print("BAG-OF-WORDS - Complete Interview Walkthrough")
    print("=" * 55)
    
    # Sample documents for demonstration
    documents = [
        "I love natural language processing",      # Doc 0
        "Machine learning is fascinating",         # Doc 1  
        "I love machine learning",                 # Doc 2
        "Natural language processing is great"     # Doc 3
    ]
    
    print("SAMPLE DOCUMENTS:")
    for i, doc in enumerate(documents):
        print(f"  Doc {i}: '{doc}'")
    
    print(f"\n" + "STEP 1: VOCABULARY BUILDING")
    print("-" * 30)
    
    # Build vocabulary and vectors
    vocab, vectors = create_bow_vector(documents)
    
    print(f"Vocabulary ({len(vocab)} words): {vocab}")
    print(f"Feature matrix: {len(vectors)} documents × {len(vocab)} features")
    
    print(f"\n" + "STEP 2: DOCUMENT VECTORS")
    print("-" * 30)
    
    # Show vectors for each document
    print("Document vectors (word counts):")
    for i, (doc, vec) in enumerate(zip(documents, vectors)):
        print(f"\nDoc {i}: '{doc}'")
        print(f"Vector: {vec}")
        
        # Show non-zero features for clarity
        non_zero_features = []
        for j, count in enumerate(vec):
            if count > 0:
                non_zero_features.append(f"{vocab[j]}:{count}")
        print(f"Non-zero: {non_zero_features}")
    
    print(f"\n" + "STEP 3: SIMILARITY CALCULATION")
    print("-" * 35)
    
    # Calculate similarity between all document pairs
    print("Pairwise document similarities:")
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  Doc {i} ↔ Doc {j}: {sim:.3f}")
            
            # Explain why this similarity makes sense
            doc_i_words = set(documents[i].lower().split())
            doc_j_words = set(documents[j].lower().split())
            overlap = len(doc_i_words & doc_j_words)
            print(f"    (Word overlap: {overlap} words)")
    
    # STEP 4: Find most similar document
    print(f"\n" + "STEP 4: SIMILARITY SEARCH")
    print("-" * 30)
    
    query_idx = 0  # Use first document as query
    most_similar = find_most_similar(documents, query_idx)
    
    print(f"Query: Doc {query_idx} - '{documents[query_idx]}'")
    print(f"Most similar: Doc {most_similar} - '{documents[most_similar]}'")
    
    # Calculate and show the similarity score
    sim_score = cosine_similarity(vectors[query_idx], vectors[most_similar])
    print(f"Similarity score: {sim_score:.3f}")
    
    # STEP 5: Vocabulary analysis
    print(f"\n" + "STEP 5: VOCABULARY ANALYSIS")
    print("-" * 35)
    
    word_doc_freq = analyze_vocabulary_distribution(documents)
    print("Word document frequencies:")
    for word, freq in sorted(word_doc_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{word}': appears in {freq}/{len(documents)} documents")
    
    print(f"\n" + "=" * 55)
    print("KEY INTERVIEW TALKING POINTS:")
    print("=" * 55)
    print("1. STRENGTHS of Bag-of-Words:")
    print("   • Simple and interpretable")
    print("   • Fast to compute")
    print("   • Works well for many classification tasks")
    print("   • Foundation for more advanced methods (TF-IDF)")
    
    print(f"\n2. LIMITATIONS of Bag-of-Words:")
    print("   • Ignores word order ('good not' vs 'not good')")
    print("   • High dimensionality (vocabulary size)")
    print("   • Sparse vectors (most elements are 0)")
    print("   • No semantic understanding ('car' vs 'automobile')")
    
    print(f"\n3. COMPLEXITY ANALYSIS:")
    print("   • Time: O(d × n × v) where d=docs, n=avg_length, v=vocab")
    print("   • Space: O(d × v) for storing count matrix")
    print("   • Similarity: O(v) for comparing two documents")
    
    print(f"\n4. PRODUCTION CONSIDERATIONS:")
    print("   • Use sparse matrices for memory efficiency")
    print("   • Consider vocabulary size limits (top-k most frequent)")
    print("   • Handle new words in test data (OOV problem)")
    print("   • Normalize vectors if document lengths vary significantly")
    
    print(f"\n" + "=" * 55)
    print("WHEN TO USE BAG-OF-WORDS:")
    print("=" * 55)
    print("✓ Quick baseline for text classification")
    print("✓ Interpretable features needed")
    print("✓ Small to medium datasets")
    print("✓ Word order not crucial for task")
    print("✗ Need semantic understanding")
    print("✗ Very large vocabulary")
    print("✗ Order/syntax matters")