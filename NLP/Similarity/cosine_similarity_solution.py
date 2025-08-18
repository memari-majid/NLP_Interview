import numpy as np
from typing import List, Tuple, Set, Dict
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import hashlib


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return text.lower().split()


def cosine_similarity(text1: str, text2: str, method='tfidf') -> float:
    """Calculate cosine similarity between two texts."""
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return sklearn_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    elif method == 'bow':  # Bag of words
        # Tokenize
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        # Create vocabulary
        vocab = set(tokens1 + tokens2)
        
        # Create vectors
        vec1 = np.array([tokens1.count(word) for word in vocab])
        vec2 = np.array([tokens2.count(word) for word in vocab])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


def jaccard_similarity(text1: str, text2: str, ngram_size: int = 1) -> float:
    """Calculate Jaccard similarity (intersection over union)."""
    if ngram_size == 1:
        # Word-level Jaccard
        set1 = set(tokenize(text1))
        set2 = set(tokenize(text2))
    else:
        # N-gram Jaccard
        set1 = set(get_ngrams(text1, ngram_size))
        set2 = set(get_ngrams(text2, ngram_size))
    
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union


def get_ngrams(text: str, n: int) -> List[str]:
    """Extract n-grams from text."""
    tokens = tokenize(text)
    ngrams = []
    
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def semantic_similarity(text1: str, text2: str, use_sentence_transformer: bool = True) -> float:
    """Calculate semantic similarity using embeddings."""
    
    if use_sentence_transformer:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode sentences
            embeddings = model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = sklearn_cosine([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except ImportError:
            print("Install sentence-transformers: pip install sentence-transformers")
            use_sentence_transformer = False
    
    if not use_sentence_transformer:
        # Fallback: Simple word embedding average
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            
            doc1 = nlp(text1)
            doc2 = nlp(text2)
            
            # Use spaCy's similarity (based on word vectors)
            return doc1.similarity(doc2)
            
        except:
            # Ultimate fallback: enhanced bag-of-words
            return cosine_similarity(text1, text2, method='tfidf')


def levenshtein_distance(text1: str, text2: str, normalize: bool = True) -> float:
    """Calculate Levenshtein (edit) distance between texts."""
    if not text1:
        return len(text2) if not normalize else 1.0
    if not text2:
        return len(text1) if not normalize else 1.0
    
    # Create matrix
    matrix = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    
    # Initialize first row and column
    for i in range(len(text1) + 1):
        matrix[i][0] = i
    for j in range(len(text2) + 1):
        matrix[0][j] = j
    
    # Fill matrix
    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i-1] == text2[j-1]:
                cost = 0
            else:
                cost = 1
            
            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # deletion
                matrix[i][j-1] + 1,      # insertion
                matrix[i-1][j-1] + cost  # substitution
            )
    
    distance = matrix[len(text1)][len(text2)]
    
    if normalize:
        # Normalize by maximum possible distance
        max_len = max(len(text1), len(text2))
        return 1 - (distance / max_len)
    
    return distance


class MinHashLSH:
    """MinHash for approximate similarity (useful for large collections)."""
    
    def __init__(self, num_hashes: int = 128, ngram_size: int = 3):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.hash_funcs = self._generate_hash_functions()
    
    def _generate_hash_functions(self):
        """Generate hash functions for MinHash."""
        hash_funcs = []
        for i in range(self.num_hashes):
            # Use different seeds for different hash functions
            seed = i
            hash_funcs.append(lambda x, s=seed: int(hashlib.md5(f"{s}{x}".encode()).hexdigest(), 16))
        return hash_funcs
    
    def get_shingles(self, text: str) -> Set[str]:
        """Convert text to shingles (n-grams)."""
        return set(get_ngrams(text, self.ngram_size))
    
    def compute_minhash(self, text: str) -> List[int]:
        """Compute MinHash signature for text."""
        shingles = self.get_shingles(text)
        if not shingles:
            return [0] * self.num_hashes
        
        signature = []
        for hash_func in self.hash_funcs:
            min_hash = min(hash_func(shingle) for shingle in shingles)
            signature.append(min_hash)
        
        return signature
    
    def jaccard_similarity_minhash(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


def find_near_duplicates(texts: List[str], threshold: float = 0.8) -> List[Tuple[int, int, float]]:
    """Find near-duplicate texts using MinHash."""
    minhash = MinHashLSH(num_hashes=64, ngram_size=3)
    
    # Compute signatures for all texts
    signatures = [minhash.compute_minhash(text) for text in texts]
    
    # Find similar pairs
    similar_pairs = []
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = minhash.jaccard_similarity_minhash(signatures[i], signatures[j])
            if similarity >= threshold:
                similar_pairs.append((i, j, similarity))
    
    return similar_pairs


def similarity_matrix(texts: List[str], method: str = 'cosine') -> np.ndarray:
    """Compute pairwise similarity matrix for multiple texts."""
    n = len(texts)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                if method == 'cosine':
                    sim = cosine_similarity(texts[i], texts[j])
                elif method == 'jaccard':
                    sim = jaccard_similarity(texts[i], texts[j])
                elif method == 'semantic':
                    sim = semantic_similarity(texts[i], texts[j])
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                matrix[i][j] = sim
                matrix[j][i] = sim  # Symmetric
    
    return matrix


if __name__ == "__main__":
    # Example 1: Compare different similarity metrics
    text1 = "The cat sat on the mat"
    text2 = "The feline rested on the rug"
    text3 = "Dogs are playing in the park"
    
    print("Text 1:", text1)
    print("Text 2:", text2)
    print("Text 3:", text3)
    print("\nSimilarity between Text 1 and Text 2:")
    
    # Different similarity metrics
    print(f"Cosine (BoW):     {cosine_similarity(text1, text2, method='bow'):.3f}")
    print(f"Cosine (TF-IDF):  {cosine_similarity(text1, text2, method='tfidf'):.3f}")
    print(f"Jaccard (words):  {jaccard_similarity(text1, text2, ngram_size=1):.3f}")
    print(f"Jaccard (3-gram): {jaccard_similarity(text1, text2, ngram_size=3):.3f}")
    print(f"Semantic:         {semantic_similarity(text1, text2, use_sentence_transformer=False):.3f}")
    print(f"Levenshtein:      {levenshtein_distance(text1, text2):.3f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Character-level similarity
    print("Character-level similarity (typos, variations):")
    word_pairs = [
        ("organize", "organise"),
        ("color", "colour"),
        ("hello", "helo"),
        ("similarity", "similarlity")
    ]
    
    for w1, w2 in word_pairs:
        dist = levenshtein_distance(w1, w2)
        print(f"{w1} <-> {w2}: {dist:.3f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Near-duplicate detection
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumped over the lazy dog",  # Very similar
        "A fast brown fox jumps over a lazy dog",        # Similar
        "Python is a great programming language",         # Different
        "Python is great for programming"                 # Somewhat similar to above
    ]
    
    print("Near-duplicate detection:")
    duplicates = find_near_duplicates(documents, threshold=0.5)
    for i, j, sim in duplicates:
        print(f"Doc {i} and Doc {j}: {sim:.3f}")
        print(f"  Doc {i}: '{documents[i]}'")
        print(f"  Doc {j}: '{documents[j]}'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Similarity matrix
    print("Similarity matrix (cosine):")
    texts = [
        "Machine learning is fascinating",
        "Deep learning is a subset of machine learning",
        "Natural language processing uses machine learning",
        "I love pizza and pasta"
    ]
    
    sim_matrix = similarity_matrix(texts, method='cosine')
    print("\nTexts:")
    for i, text in enumerate(texts):
        print(f"{i}: {text}")
    
    print("\nSimilarity matrix:")
    print(sim_matrix.round(2))
