import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFFromScratch:
    """TF-IDF implementation from scratch for educational purposes."""
    
    def __init__(self, use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}
        self.idf_ = None
        self.documents = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def fit(self, documents: List[str]):
        """Fit the TF-IDF model."""
        self.documents = documents
        n_docs = len(documents)
        
        # Build vocabulary
        vocab_set = set()
        doc_freq = Counter()
        
        tokenized_docs = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            vocab_set.update(unique_tokens)
            doc_freq.update(unique_tokens)
        
        # Create vocabulary mapping
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocabulary_)
        
        # Calculate IDF
        if self.use_idf:
            self.idf_ = np.zeros(vocab_size)
            for word, idx in self.vocabulary_.items():
                df = doc_freq.get(word, 0)
                if self.smooth_idf:
                    self.idf_[idx] = math.log((n_docs + 1) / (df + 1)) + 1
                else:
                    self.idf_[idx] = math.log(n_docs / df) + 1
        else:
            self.idf_ = np.ones(vocab_size)
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF matrix."""
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)
            
            # Calculate TF
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    word_idx = self.vocabulary_[token]
                    
                    if self.sublinear_tf:
                        tf = 1 + math.log(count)
                    else:
                        tf = count
                    
                    tfidf_matrix[doc_idx, word_idx] = tf * self.idf_[word_idx]
            
            # L2 normalization
            norm = np.linalg.norm(tfidf_matrix[doc_idx])
            if norm > 0:
                tfidf_matrix[doc_idx] /= norm
        
        return tfidf_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


def calculate_tfidf(documents: List[str], method='sklearn') -> np.ndarray:
    """Calculate TF-IDF matrix using specified method."""
    if method == 'sklearn':
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(documents).toarray()
    else:  # from_scratch
        tfidf = TFIDFFromScratch()
        return tfidf.fit_transform(documents)


def find_similar_documents(query: str, documents: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
    """Find top-k similar documents to query using TF-IDF + cosine similarity."""
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit on documents and transform both documents and query
    doc_tfidf = vectorizer.fit_transform(documents)
    query_tfidf = vectorizer.transform([query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_tfidf, doc_tfidf)[0]
    
    # Get top-k similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [(idx, similarities[idx]) for idx in top_indices]
    return results


def compare_tfidf_methods(documents: List[str]) -> Dict[str, np.ndarray]:
    """Compare different TF-IDF implementations."""
    results = {}
    
    # Sklearn default
    vectorizer = TfidfVectorizer()
    results['sklearn_default'] = vectorizer.fit_transform(documents).toarray()
    
    # From scratch
    tfidf_scratch = TFIDFFromScratch()
    results['from_scratch'] = tfidf_scratch.fit_transform(documents)
    
    # Sklearn with different parameters
    vectorizer_sublinear = TfidfVectorizer(sublinear_tf=True)
    results['sklearn_sublinear'] = vectorizer_sublinear.fit_transform(documents).toarray()
    
    # Without IDF (just term frequency)
    vectorizer_no_idf = TfidfVectorizer(use_idf=False)
    results['tf_only'] = vectorizer_no_idf.fit_transform(documents).toarray()
    
    return results


class BM25:
    """BM25 ranking function for comparison with TF-IDF."""
    
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.n_docs = 0
        self.vocabulary = set()
    
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def fit(self, documents: List[str]):
        """Fit BM25 model."""
        self.n_docs = len(documents)
        
        # Calculate document frequencies and lengths
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            
            unique_tokens = set(tokens)
            self.vocabulary.update(unique_tokens)
            
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # Calculate IDF
        for word, freq in self.doc_freqs.items():
            self.idf[word] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5))
        
        return self
    
    def score(self, query: str, document: str, doc_idx: int) -> float:
        """Calculate BM25 score for query-document pair."""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)
        doc_token_freqs = Counter(doc_tokens)
        
        score = 0.0
        doc_len = self.doc_len[doc_idx]
        
        for token in query_tokens:
            if token not in self.vocabulary:
                continue
                
            freq = doc_token_freqs.get(token, 0)
            idf = self.idf.get(token, 0)
            
            numerator = idf * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += numerator / denominator
        
        return score
    
    def rank_documents(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """Rank documents using BM25."""
        scores = []
        
        for idx, doc in enumerate(documents):
            score = self.score(query, doc, idx)
            scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


def demonstrate_tfidf_properties():
    """Demonstrate key properties of TF-IDF."""
    # Example 1: TF-IDF favors rare terms
    docs_rare_terms = [
        "The quick brown fox",
        "The quick brown dog", 
        "The unique purple elephant"  # 'unique', 'purple', 'elephant' are rare
    ]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_rare_terms)
    feature_names = vectorizer.get_feature_names_out()
    
    print("TF-IDF scores showing rare terms get higher weights:")
    for doc_idx, doc in enumerate(docs_rare_terms):
        print(f"\nDocument {doc_idx}: '{doc}'")
        doc_tfidf = tfidf_matrix[doc_idx].toarray()[0]
        word_scores = [(feature_names[i], doc_tfidf[i]) 
                      for i in range(len(feature_names)) if doc_tfidf[i] > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        for word, score in word_scores:
            print(f"  {word}: {score:.3f}")


if __name__ == "__main__":
    # Example 1: Basic TF-IDF
    documents = [
        "The cat sat on the mat",
        "The dog sat on the log", 
        "Cats and dogs are pets",
        "The mat was comfortable"
    ]
    
    print("TF-IDF Matrix (sklearn):")
    tfidf_matrix = calculate_tfidf(documents, method='sklearn')
    print(tfidf_matrix.shape)
    print(tfidf_matrix[:2])  # First 2 documents
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Document similarity
    query = "cat on mat"
    print(f"Query: '{query}'")
    print("Similar documents:")
    similar_docs = find_similar_documents(query, documents, top_k=3)
    for idx, score in similar_docs:
        print(f"  Doc {idx}: '{documents[idx]}' (similarity: {score:.3f})")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Compare with BM25
    print("BM25 Ranking for same query:")
    bm25 = BM25()
    bm25.fit(documents)
    bm25_results = bm25.rank_documents(query, documents, top_k=3)
    for idx, score in bm25_results:
        print(f"  Doc {idx}: '{documents[idx]}' (BM25 score: {score:.3f})")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Demonstrate TF-IDF properties
    demonstrate_tfidf_properties()
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Compare methods
    print("Comparing TF-IDF methods (first doc, first 5 features):")
    comparisons = compare_tfidf_methods(documents[:2])
    for method, matrix in comparisons.items():
        print(f"\n{method}:")
        print(matrix[0][:5])  # First 5 features of first document
