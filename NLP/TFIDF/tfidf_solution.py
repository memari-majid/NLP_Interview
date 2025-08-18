import math
from typing import List, Dict
from collections import Counter

def compute_tfidf(documents: List[str]) -> List[Dict[str, float]]:
    """
    Compute TF-IDF vectors for documents.
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    - Emphasizes important words that appear frequently in a document
    - But rarely across the entire collection
    """
    # STEP 1: Handle edge cases
    if not documents:
        return []
    
    # STEP 2: Tokenize all documents (simple whitespace splitting)
    # In real interviews, discuss more sophisticated tokenization
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # STEP 3: Build vocabulary from all unique words
    # This creates our feature space - each unique word becomes a dimension
    vocab = set()
    for doc in tokenized_docs:
        vocab.update(doc)
    vocab = sorted(list(vocab))  # Sort for consistency
    
    # STEP 4: Calculate Document Frequency (DF) for each term
    # DF = number of documents containing the term
    # Used in IDF calculation: IDF = log(total_docs / doc_freq)
    doc_freq = {}
    for term in vocab:
        doc_freq[term] = sum(1 for doc in tokenized_docs if term in doc)
    
    # STEP 5: Calculate TF-IDF for each document
    tfidf_vectors = []
    num_docs = len(documents)
    
    for doc in tokenized_docs:
        # Count term frequencies in this document
        term_counts = Counter(doc)
        doc_length = len(doc)
        tfidf_vector = {}
        
        for term in vocab:
            # TERM FREQUENCY (TF): How often term appears in this document
            # Normalized by document length to handle different document sizes
            tf = term_counts[term] / doc_length if doc_length > 0 else 0
            
            # INVERSE DOCUMENT FREQUENCY (IDF): How rare the term is across collection
            # log(total_docs / docs_containing_term)
            # Rare terms get higher IDF scores
            idf = math.log(num_docs / doc_freq[term])
            
            # TF-IDF SCORE: Combines term importance in document (TF) 
            # with term rarity in collection (IDF)
            tfidf_vector[term] = tf * idf
        
        tfidf_vectors.append(tfidf_vector)
    
    return tfidf_vectors

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Calculate cosine similarity between two TF-IDF vectors.
    
    Cosine similarity = dot_product(v1, v2) / (||v1|| × ||v2||)
    - Returns value between 0 and 1 (since TF-IDF values are non-negative)
    - 1.0 = identical vectors, 0.0 = completely different
    """
    # STEP 1: Find common terms between the two vectors
    # Only these contribute to the dot product
    common_terms = set(vec1.keys()) & set(vec2.keys())
    
    if not common_terms:
        return 0.0  # No overlap = no similarity
    
    # STEP 2: Calculate dot product
    # Sum of element-wise multiplication for common terms
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    
    # STEP 3: Calculate vector norms (magnitudes)
    # ||v|| = sqrt(sum of squared elements)
    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    # STEP 4: Handle edge case of zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Can't compute similarity with zero vector
    
    # STEP 5: Return cosine similarity
    # Normalized dot product gives us the cosine of angle between vectors
    return dot_product / (norm1 * norm2)

def find_similar_documents(documents: List[str], query: str) -> int:
    """
    Find most similar document to query using TF-IDF + cosine similarity.
    
    This is the core of many search and recommendation systems.
    """
    # STEP 1: Create unified document collection
    # Include query as the last document for TF-IDF calculation
    all_texts = documents + [query]
    
    # STEP 2: Compute TF-IDF for all documents + query
    # This ensures query and documents are in same vector space
    tfidf_vectors = compute_tfidf(all_texts)
    
    # STEP 3: Extract query vector (last one)
    query_vector = tfidf_vectors[-1]
    
    # STEP 4: Compare query with each document
    max_similarity = -1
    most_similar_idx = 0
    
    # Only compare with documents (exclude query vector)
    for i, doc_vector in enumerate(tfidf_vectors[:-1]):
        similarity = cosine_similarity(query_vector, doc_vector)
        
        # Track the most similar document
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_idx = i
    
    return most_similar_idx

# INTERVIEW DEMO: Show step-by-step execution
if __name__ == "__main__":
    print("TF-IDF Implementation - Step by Step")
    print("=" * 50)
    
    # Sample documents (what you might discuss in interview)
    docs = [
        "cat sat on mat",      # Document 0
        "dog sat on log",      # Document 1  
        "cat and dog are pets" # Document 2
    ]
    
    query = "cat on mat"
    
    print("DOCUMENTS:")
    for i, doc in enumerate(docs):
        print(f"  {i}: '{doc}'")
    print(f"QUERY: '{query}'")
    
    print("\n" + "=" * 50)
    print("STEP-BY-STEP TF-IDF CALCULATION")
    print("=" * 50)
    
    # Manual walkthrough for interview explanation
    all_words = set()
    for doc in docs + [query]:
        all_words.update(doc.split())
    vocab = sorted(list(all_words))
    
    print(f"1. VOCABULARY: {vocab}")
    
    # Document frequencies
    doc_freq = {}
    for word in vocab:
        df = sum(1 for doc in docs if word in doc.split())
        doc_freq[word] = df
    
    print(f"\n2. DOCUMENT FREQUENCIES:")
    for word, df in doc_freq.items():
        idf = math.log(len(docs) / df)
        print(f"   '{word}': appears in {df}/{len(docs)} docs, IDF = {idf:.3f}")
    
    # TF-IDF calculation
    print(f"\n3. TF-IDF CALCULATION FOR FIRST DOCUMENT:")
    doc0_words = docs[0].split()
    doc0_counts = Counter(doc0_words)
    doc0_length = len(doc0_words)
    
    for word in sorted(doc0_counts.keys()):
        tf = doc0_counts[word] / doc0_length
        idf = math.log(len(docs) / doc_freq[word])
        tfidf = tf * idf
        print(f"   '{word}': TF={tf:.3f}, IDF={idf:.3f}, TF-IDF={tfidf:.3f}")
    
    # Full computation
    print(f"\n4. SIMILARITY COMPUTATION:")
    tfidf_vectors = compute_tfidf(docs)
    
    print(f"   Query '{query}' TF-IDF scores:")
    query_tfidf = compute_tfidf([query])[0]
    for word, score in sorted(query_tfidf.items()):
        if score > 0:
            print(f"     '{word}': {score:.3f}")
    
    # Find most similar
    similar_idx = find_similar_documents(docs, query)
    print(f"\n5. RESULT:")
    print(f"   Most similar document: {similar_idx}")
    print(f"   Document: '{docs[similar_idx]}'")
    
    # Show why it's most similar
    query_vec = compute_tfidf(docs + [query])[-1]
    for i, doc in enumerate(docs):
        doc_vec = tfidf_vectors[i]
        sim = cosine_similarity(query_vec, doc_vec)
        print(f"   Similarity with doc {i}: {sim:.3f}")
    
    print("\n" + "=" * 50)
    print("KEY INTERVIEW POINTS TO MENTION:")
    print("• TF-IDF balances term frequency with document rarity")
    print("• Cosine similarity measures angle between vectors (not magnitude)")
    print("• Time complexity: O(d×v) where d=docs, v=vocab_size")
    print("• Space complexity: O(d×v) for storing TF-IDF matrix")
    print("• Scaling: Use sparse matrices for large collections")
    print("• Alternative: BM25 for better relevance scoring")