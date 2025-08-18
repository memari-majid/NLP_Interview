import math
from typing import List, Dict
from collections import Counter

def compute_tfidf(documents: List[str]) -> List[Dict[str, float]]:
    """Compute TF-IDF vectors for documents."""
    if not documents:
        return []
    
    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Build vocabulary
    vocab = set()
    for doc in tokenized_docs:
        vocab.update(doc)
    
    # Calculate document frequency for each term
    doc_freq = {}
    for term in vocab:
        doc_freq[term] = sum(1 for doc in tokenized_docs if term in doc)
    
    # Calculate TF-IDF for each document
    tfidf_vectors = []
    num_docs = len(documents)
    
    for doc in tokenized_docs:
        term_counts = Counter(doc)
        doc_length = len(doc)
        tfidf_vector = {}
        
        for term in vocab:
            tf = term_counts[term] / doc_length if doc_length > 0 else 0
            idf = math.log(num_docs / doc_freq[term])
            tfidf_vector[term] = tf * idf
        
        tfidf_vectors.append(tfidf_vector)
    
    return tfidf_vectors

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Calculate cosine similarity between two TF-IDF vectors."""
    # Get common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())
    
    if not common_terms:
        return 0.0
    
    # Calculate dot product and norms
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def find_similar_documents(documents: List[str], query: str) -> int:
    """Find most similar document to query."""
    # Compute TF-IDF for documents + query
    all_texts = documents + [query]
    tfidf_vectors = compute_tfidf(all_texts)
    
    query_vector = tfidf_vectors[-1]  # Last vector is the query
    
    # Find most similar document
    max_similarity = -1
    most_similar_idx = 0
    
    for i, doc_vector in enumerate(tfidf_vectors[:-1]):  # Exclude query vector
        similarity = cosine_similarity(query_vector, doc_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_idx = i
    
    return most_similar_idx

# Test
if __name__ == "__main__":
    docs = [
        "cat sat on mat", 
        "dog sat on log", 
        "cat and dog are pets"
    ]
    
    query = "cat on mat"
    
    # Test TF-IDF computation
    tfidf_vectors = compute_tfidf(docs)
    print(f"TF-IDF for '{docs[0]}':")
    for term, score in list(tfidf_vectors[0].items())[:3]:
        print(f"  {term}: {score:.3f}")
    
    # Test similarity
    similar_idx = find_similar_documents(docs, query)
    print(f"\nMost similar to '{query}': Document {similar_idx}")
    print(f"'{docs[similar_idx]}'")