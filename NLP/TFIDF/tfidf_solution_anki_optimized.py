"""
TF-IDF Implementation - Optimized for Anki Memorization
========================================================
KEY FORMULA: TF-IDF(t,d) = TF(t,d) × IDF(t)
"""

import math
from typing import List, Dict
from collections import Counter

# ============= CORE IMPLEMENTATION TO MEMORIZE =============

def compute_tfidf(documents: List[str]) -> List[Dict[str, float]]:
    """
    MEMORIZE THIS IMPLEMENTATION:
    1. Tokenize documents
    2. Build vocabulary
    3. Calculate document frequency
    4. Compute TF-IDF for each doc
    """
    if not documents:
        return []
    
    # 1. TOKENIZE
    tokenized = [doc.lower().split() for doc in documents]
    
    # 2. BUILD VOCAB
    vocab = set()
    for doc in tokenized:
        vocab.update(doc)
    vocab = sorted(list(vocab))
    
    # 3. DOCUMENT FREQUENCY
    doc_freq = {}
    for term in vocab:
        doc_freq[term] = sum(1 for doc in tokenized if term in doc)
    
    # 4. COMPUTE TF-IDF
    tfidf_vectors = []
    N = len(documents)
    
    for doc in tokenized:
        counts = Counter(doc)
        doc_len = len(doc)
        vector = {}
        
        for term in vocab:
            # KEY FORMULAS TO MEMORIZE:
            tf = counts[term] / doc_len if doc_len > 0 else 0
            idf = math.log(N / doc_freq[term])
            vector[term] = tf * idf
        
        tfidf_vectors.append(vector)
    
    return tfidf_vectors

# ============= INTERVIEW TALKING POINTS =============
"""
MEMORIZE THESE KEY POINTS:

1. TF (Term Frequency):
   - Formula: tf = count(term in doc) / total_terms_in_doc
   - Normalizes for document length
   - Higher value = term appears more in this doc

2. IDF (Inverse Document Frequency):
   - Formula: idf = log(N / df)
   - N = total documents
   - df = documents containing term
   - Higher value = term is rarer across corpus

3. Why use log in IDF?
   - Dampens effect of very rare terms
   - Makes scale more manageable
   - Natural for information theory

4. Complexity:
   - TIME: O(d × v) where d=docs, v=vocab_size
   - SPACE: O(d × v) for TF-IDF matrix

5. Common Mistakes:
   - Forgetting to normalize TF
   - Not handling empty documents
   - Division by zero in IDF
   - Using wrong log base (natural log is standard)

6. Improvements:
   - Use sparse matrices for large vocab
   - Consider BM25 for better ranking
   - Add smoothing for unseen terms
"""

# ============= COSINE SIMILARITY (BONUS) =============

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    MEMORIZE: cosine_sim = dot_product / (norm1 × norm2)
    """
    # Find common terms
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    
    # Calculate components
    dot_product = sum(vec1[t] * vec2[t] for t in common)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

# ============= QUICK TEST (Interview Demo) =============

if __name__ == "__main__":
    # MEMORIZE THIS EXAMPLE:
    docs = [
        "cat sat mat",      # Doc 0
        "dog sat log",      # Doc 1
        "cat dog pets"      # Doc 2
    ]
    
    tfidf = compute_tfidf(docs)
    
    print("TF-IDF EXAMPLE TO MEMORIZE:")
    print("Documents:", docs)
    print("\nVocabulary: cat, dog, log, mat, pets, sat")
    print("\nDocument Frequencies:")
    print("  cat: 2 docs → IDF = log(3/2) = 0.405")
    print("  mat: 1 doc  → IDF = log(3/1) = 1.099")
    print("  sat: 2 docs → IDF = log(3/2) = 0.405")
    
    print("\nTF-IDF for 'cat sat mat':")
    print("  cat: TF=1/3 × IDF=0.405 = 0.135")
    print("  sat: TF=1/3 × IDF=0.405 = 0.135")
    print("  mat: TF=1/3 × IDF=1.099 = 0.366")