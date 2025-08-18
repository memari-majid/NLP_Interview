import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import Counter, defaultdict
import re
import math


class BagOfWords:
    """Bag of Words vectorizer implemented from scratch."""
    
    def __init__(self, max_features: Optional[int] = None, binary: bool = False,
                 min_df: Union[int, float] = 1, max_df: Union[int, float] = 1.0,
                 token_pattern: str = r'\b\w+\b', lowercase: bool = True,
                 stop_words: Optional[Set[str]] = None, ngram_range: Tuple[int, int] = (1, 1)):
        """
        Initialize Bag of Words vectorizer.
        
        Args:
            max_features: Maximum number of features to keep
            binary: If True, use binary (0/1) counts instead of frequencies
            min_df: Minimum document frequency (int) or proportion (float)
            max_df: Maximum document frequency (int) or proportion (float)
            token_pattern: Regex pattern for tokenization
            lowercase: Convert text to lowercase
            stop_words: Set of stop words to ignore
            ngram_range: Range of n-grams to extract (min_n, max_n)
        """
        self.max_features = max_features
        self.binary = binary
        self.min_df = min_df
        self.max_df = max_df
        self.token_pattern = re.compile(token_pattern)
        self.lowercase = lowercase
        self.stop_words = stop_words or set()
        self.ngram_range = ngram_range
        
        # Fitted attributes
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.document_frequencies_ = {}
        self.n_docs_fitted_ = 0
        self.is_fitted_ = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using regex pattern."""
        if self.lowercase:
            text = text.lower()
        
        tokens = self.token_pattern.findall(text)
        
        # Remove stop words
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Extract n-grams from tokens."""
        ngrams = []
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def _build_vocabulary(self, documents: List[str]):
        """Build vocabulary from documents."""
        # Count document frequencies
        doc_frequencies = defaultdict(int)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            
            # Get unique n-grams in this document
            unique_ngrams = set(ngrams)
            
            for ngram in unique_ngrams:
                doc_frequencies[ngram] += 1
        
        # Apply min_df and max_df filtering
        n_docs = len(documents)
        min_df_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_df_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        
        # Filter vocabulary
        filtered_vocab = {}
        for ngram, freq in doc_frequencies.items():
            if min_df_count <= freq <= max_df_count:
                filtered_vocab[ngram] = freq
        
        # Sort by document frequency (most common first) for consistency
        sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
        
        # Apply max_features limit
        if self.max_features:
            sorted_vocab = sorted_vocab[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(sorted_vocab)}
        self.feature_names_ = [term for term, _ in sorted_vocab]
        self.document_frequencies_ = {term: freq for term, freq in sorted_vocab}
        self.n_docs_fitted_ = n_docs
    
    def fit(self, documents: List[str]):
        """Fit the vectorizer on documents."""
        if not documents:
            raise ValueError("Cannot fit on empty document list")
        
        self._build_vocabulary(documents)
        self.is_fitted_ = True
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to bag-of-words vectors."""
        if not self.is_fitted_:
            raise ValueError("Vectorizer has not been fitted yet")
        
        if not documents:
            return np.zeros((0, len(self.vocabulary_)))
        
        # Initialize matrix
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.int32 if not self.binary else np.bool8)
        
        # Transform each document
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            
            if self.binary:
                # Binary representation (presence/absence)
                unique_ngrams = set(ngrams)
                for ngram in unique_ngrams:
                    if ngram in self.vocabulary_:
                        X[doc_idx, self.vocabulary_[ngram]] = 1
            else:
                # Count representation
                ngram_counts = Counter(ngrams)
                for ngram, count in ngram_counts.items():
                    if ngram in self.vocabulary_:
                        X[doc_idx, self.vocabulary_[ngram]] = count
        
        return X
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform documents."""
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)."""
        if not self.is_fitted_:
            raise ValueError("Vectorizer has not been fitted yet")
        return self.feature_names_.copy()
    
    def get_vocabulary_stats(self) -> Dict[str, Union[int, float]]:
        """Get vocabulary statistics."""
        if not self.is_fitted_:
            raise ValueError("Vectorizer has not been fitted yet")
        
        return {
            'vocabulary_size': len(self.vocabulary_),
            'avg_doc_frequency': np.mean(list(self.document_frequencies_.values())),
            'max_doc_frequency': max(self.document_frequencies_.values()),
            'min_doc_frequency': min(self.document_frequencies_.values()),
            'total_documents': self.n_docs_fitted_
        }
    
    def get_top_features(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get top features by document frequency."""
        if not self.is_fitted_:
            raise ValueError("Vectorizer has not been fitted yet")
        
        sorted_features = sorted(
            self.document_frequencies_.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:top_k]


class CharacterBagOfWords:
    """Character-level Bag of Words implementation."""
    
    def __init__(self, max_features: Optional[int] = None, binary: bool = False,
                 char_range: Tuple[int, int] = (1, 3), lowercase: bool = True):
        """
        Initialize Character-level BoW.
        
        Args:
            max_features: Maximum number of character n-grams to keep
            binary: Use binary representation
            char_range: Range of character n-grams (min_n, max_n)
            lowercase: Convert to lowercase
        """
        self.max_features = max_features
        self.binary = binary
        self.char_range = char_range
        self.lowercase = lowercase
        
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.is_fitted_ = False
    
    def _get_char_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text."""
        if self.lowercase:
            text = text.lower()
        
        char_ngrams = []
        min_n, max_n = self.char_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(text) - n + 1):
                char_ngram = text[i:i+n]
                char_ngrams.append(char_ngram)
        
        return char_ngrams
    
    def fit(self, documents: List[str]):
        """Fit on character n-grams."""
        char_ngram_counts = Counter()
        
        for doc in documents:
            char_ngrams = self._get_char_ngrams(doc)
            char_ngram_counts.update(char_ngrams)
        
        # Select top features
        if self.max_features:
            top_ngrams = char_ngram_counts.most_common(self.max_features)
        else:
            top_ngrams = list(char_ngram_counts.items())
        
        self.vocabulary_ = {ngram: idx for idx, (ngram, _) in enumerate(top_ngrams)}
        self.feature_names_ = [ngram for ngram, _ in top_ngrams]
        self.is_fitted_ = True
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform to character n-gram vectors."""
        if not self.is_fitted_:
            raise ValueError("Vectorizer has not been fitted yet")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.int32)
        
        for doc_idx, doc in enumerate(documents):
            char_ngrams = self._get_char_ngrams(doc)
            
            if self.binary:
                unique_ngrams = set(char_ngrams)
                for ngram in unique_ngrams:
                    if ngram in self.vocabulary_:
                        X[doc_idx, self.vocabulary_[ngram]] = 1
            else:
                ngram_counts = Counter(char_ngrams)
                for ngram, count in ngram_counts.items():
                    if ngram in self.vocabulary_:
                        X[doc_idx, self.vocabulary_[ngram]] = count
        
        return X
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform."""
        return self.fit(documents).transform(documents)


class FeatureSelector:
    """Feature selection methods for BoW vectors."""
    
    @staticmethod
    def chi_square(X: np.ndarray, y: np.ndarray, k: int = 10) -> List[int]:
        """Select top k features using chi-square test."""
        n_features = X.shape[1]
        chi2_scores = []
        
        # Convert to binary classification for simplicity
        y_binary = (y > 0).astype(int)
        
        for feature_idx in range(n_features):
            feature = X[:, feature_idx]
            
            # Create contingency table
            # feature_present & class_1, feature_present & class_0
            # feature_absent & class_1, feature_absent & class_0
            
            present_and_pos = np.sum((feature > 0) & (y_binary == 1))
            present_and_neg = np.sum((feature > 0) & (y_binary == 0))
            absent_and_pos = np.sum((feature == 0) & (y_binary == 1))
            absent_and_neg = np.sum((feature == 0) & (y_binary == 0))
            
            # Chi-square statistic
            observed = np.array([[present_and_pos, present_and_neg],
                               [absent_and_pos, absent_and_neg]])
            
            row_totals = observed.sum(axis=1)
            col_totals = observed.sum(axis=0)
            total = observed.sum()
            
            if total == 0:
                chi2_scores.append(0)
                continue
            
            expected = np.outer(row_totals, col_totals) / total
            
            # Avoid division by zero
            expected = np.where(expected == 0, 1e-10, expected)
            
            chi2_stat = np.sum((observed - expected) ** 2 / expected)
            chi2_scores.append(chi2_stat)
        
        # Get top k features
        top_indices = np.argsort(chi2_scores)[-k:][::-1]
        return top_indices.tolist()
    
    @staticmethod
    def mutual_information(X: np.ndarray, y: np.ndarray, k: int = 10) -> List[int]:
        """Select top k features using mutual information."""
        n_features = X.shape[1]
        mi_scores = []
        
        for feature_idx in range(n_features):
            feature = X[:, feature_idx] > 0  # Binarize feature
            
            # Calculate mutual information
            mi = FeatureSelector._calculate_mi(feature, y)
            mi_scores.append(mi)
        
        top_indices = np.argsort(mi_scores)[-k:][::-1]
        return top_indices.tolist()
    
    @staticmethod
    def _calculate_mi(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two variables."""
        # Joint distribution
        joint_counts = defaultdict(int)
        for xi, yi in zip(x, y):
            joint_counts[(xi, yi)] += 1
        
        # Marginal distributions
        x_counts = Counter(x)
        y_counts = Counter(y)
        
        n_samples = len(x)
        mi = 0.0
        
        for (xi, yi), joint_count in joint_counts.items():
            p_joint = joint_count / n_samples
            p_x = x_counts[xi] / n_samples
            p_y = y_counts[yi] / n_samples
            
            if p_joint > 0 and p_x > 0 and p_y > 0:
                mi += p_joint * math.log2(p_joint / (p_x * p_y))
        
        return mi


def compare_bow_approaches(documents: List[str], labels: List[int]):
    """Compare different BoW approaches."""
    results = {}
    
    # Word-level BoW
    word_bow = BagOfWords()
    X_word = word_bow.fit_transform(documents)
    
    # Character-level BoW
    char_bow = CharacterBagOfWords(char_range=(2, 4))
    X_char = char_bow.fit_transform(documents)
    
    # N-gram BoW
    ngram_bow = BagOfWords(ngram_range=(1, 2))
    X_ngram = ngram_bow.fit_transform(documents)
    
    results = {
        'word_bow': {
            'shape': X_word.shape,
            'vocabulary_size': len(word_bow.vocabulary_),
            'sparsity': 1 - np.count_nonzero(X_word) / X_word.size
        },
        'char_bow': {
            'shape': X_char.shape,
            'vocabulary_size': len(char_bow.vocabulary_),
            'sparsity': 1 - np.count_nonzero(X_char) / X_char.size
        },
        'ngram_bow': {
            'shape': X_ngram.shape,
            'vocabulary_size': len(ngram_bow.vocabulary_),
            'sparsity': 1 - np.count_nonzero(X_ngram) / X_ngram.size
        }
    }
    
    return results


# Demo and test functions
def create_sample_documents():
    """Create sample documents for testing."""
    documents = [
        "I love natural language processing",
        "Machine learning is fascinating",
        "Deep learning uses neural networks", 
        "I enjoy programming and coding",
        "Natural language processing is part of AI",
        "Python is great for machine learning",
        "Neural networks can process language",
        "I love coding in Python",
        "AI and machine learning are related",
        "Programming is fun and rewarding"
    ]
    
    # Simple binary labels (positive/negative sentiment)
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # All positive for demo
    
    return documents, labels


if __name__ == "__main__":
    print("Bag of Words Implementation from Scratch\n")
    
    # Create sample data
    documents, labels = create_sample_documents()
    
    print(f"Sample Documents ({len(documents)} docs):")
    for i, doc in enumerate(documents[:3]):
        print(f"  {i+1}. {doc}")
    print("  ...")
    
    print("\n" + "="*60 + "\n")
    
    # Basic BoW
    print("Basic Bag of Words:")
    bow = BagOfWords()
    X_bow = bow.fit_transform(documents)
    
    print(f"Vocabulary size: {len(bow.vocabulary_)}")
    print(f"Feature matrix shape: {X_bow.shape}")
    print(f"Top features: {bow.get_top_features(5)}")
    
    # Show first document vector
    print(f"\nFirst document: '{documents[0]}'")
    print("Feature vector (first 10 features):")
    feature_names = bow.get_feature_names()
    for i in range(min(10, len(feature_names))):
        if X_bow[0, i] > 0:
            print(f"  {feature_names[i]}: {X_bow[0, i]}")
    
    print("\n" + "="*60 + "\n")
    
    # Binary BoW
    print("Binary Bag of Words:")
    binary_bow = BagOfWords(binary=True)
    X_binary = binary_bow.fit_transform(documents)
    
    print(f"Binary matrix shape: {X_binary.shape}")
    print(f"First document binary vector (first 10): {X_binary[0, :10]}")
    
    print("\n" + "="*60 + "\n")
    
    # BoW with constraints
    print("Constrained Bag of Words:")
    
    # Add some stop words
    stop_words = {"is", "are", "and", "the", "in", "for"}
    
    constrained_bow = BagOfWords(
        max_features=20, 
        min_df=2, 
        stop_words=stop_words
    )
    X_constrained = constrained_bow.fit_transform(documents)
    
    print(f"Constrained vocabulary size: {len(constrained_bow.vocabulary_)}")
    print("Vocabulary (after filtering):")
    for feature in constrained_bow.get_feature_names():
        print(f"  {feature}")
    
    stats = constrained_bow.get_vocabulary_stats()
    print(f"\nVocabulary stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60 + "\n")
    
    # N-gram BoW
    print("N-gram Bag of Words:")
    ngram_bow = BagOfWords(ngram_range=(1, 2), max_features=30)
    X_ngram = ngram_bow.fit_transform(documents)
    
    print(f"N-gram vocabulary size: {len(ngram_bow.vocabulary_)}")
    print("Sample bigrams:")
    features = ngram_bow.get_feature_names()
    bigrams = [f for f in features if ' ' in f]
    for bigram in bigrams[:10]:
        print(f"  {bigram}")
    
    print("\n" + "="*60 + "\n")
    
    # Character-level BoW
    print("Character-level Bag of Words:")
    char_bow = CharacterBagOfWords(char_range=(2, 4), max_features=50)
    X_char = char_bow.fit_transform(documents)
    
    print(f"Character n-gram matrix shape: {X_char.shape}")
    print("Sample character n-grams:")
    char_features = char_bow.feature_names_[:20]
    for feature in char_features:
        print(f"  '{feature}'")
    
    print("\n" + "="*60 + "\n")
    
    # Feature Selection
    print("Feature Selection:")
    
    # Create some class labels for feature selection
    # (positive if contains "learning" or "programming")
    feature_labels = []
    for doc in documents:
        if "learning" in doc.lower() or "programming" in doc.lower():
            feature_labels.append(1)
        else:
            feature_labels.append(0)
    
    feature_labels = np.array(feature_labels)
    
    # Chi-square feature selection
    top_chi2_features = FeatureSelector.chi_square(X_bow, feature_labels, k=5)
    
    print("Top 5 features (Chi-square):")
    for idx in top_chi2_features:
        feature_name = bow.get_feature_names()[idx]
        print(f"  {feature_name} (index: {idx})")
    
    # Mutual Information feature selection
    top_mi_features = FeatureSelector.mutual_information(X_bow, feature_labels, k=5)
    
    print("\nTop 5 features (Mutual Information):")
    for idx in top_mi_features:
        feature_name = bow.get_feature_names()[idx]
        print(f"  {feature_name} (index: {idx})")
    
    print("\n" + "="*60 + "\n")
    
    # Comparison of approaches
    print("Comparison of BoW Approaches:")
    comparison = compare_bow_approaches(documents, labels)
    
    for approach, metrics in comparison.items():
        print(f"\n{approach.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\n" + "="*60 + "\n")
    
    # Out-of-vocabulary handling
    print("Out-of-Vocabulary (OOV) Handling:")
    
    # Train on subset
    train_docs = documents[:5]
    test_docs = ["Quantum computing is revolutionary", "I hate bugs in code"]
    
    oov_bow = BagOfWords()
    oov_bow.fit(train_docs)
    
    print("Training vocabulary:", oov_bow.get_feature_names()[:10])
    
    X_test = oov_bow.transform(test_docs)
    print(f"\nTest documents shape: {X_test.shape}")
    print("Test document 1 vector (non-zero elements):")
    
    feature_names = oov_bow.get_feature_names()
    for i, count in enumerate(X_test[0]):
        if count > 0:
            print(f"  {feature_names[i]}: {count}")
    
    print("\nNote: Words like 'quantum', 'revolutionary' are ignored (OOV)")
    
    print("\n" + "="*60 + "\n")
    print("Summary:")
    print("✓ Basic BoW: Simple word counting")
    print("✓ Binary BoW: Presence/absence representation") 
    print("✓ Constrained BoW: Vocabulary filtering")
    print("✓ N-gram BoW: Captures word combinations")
    print("✓ Character BoW: Handles morphology/typos")
    print("✓ Feature Selection: Identifies most informative features")
    print("✓ OOV Handling: Gracefully ignores unknown words")
