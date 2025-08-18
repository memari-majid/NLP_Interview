import numpy as np
from typing import List, Tuple, Dict, Union
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class LSAModel:
    """Latent Semantic Analysis using SVD."""
    
    def __init__(self, num_topics: int = 5, use_tfidf: bool = True):
        self.num_topics = num_topics
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.svd = None
        self.document_topic_matrix = None
        self.topic_word_matrix = None
        self.vocabulary = None
    
    def fit(self, documents: List[str]):
        """Fit LSA model to documents."""
        # Vectorize documents
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.vocabulary = self.vectorizer.get_feature_names_out()
        
        # Apply SVD
        self.svd = TruncatedSVD(n_components=self.num_topics, random_state=42)
        self.document_topic_matrix = self.svd.fit_transform(doc_term_matrix)
        
        # Topic-word matrix (V^T in SVD)
        self.topic_word_matrix = self.svd.components_
        
        # Normalize for interpretation
        self.document_topic_matrix = normalize(self.document_topic_matrix, axis=1)
    
    def get_topics(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """Extract top words for each topic."""
        topics = []
        
        for topic_idx in range(self.num_topics):
            # Get word scores for this topic
            word_scores = self.topic_word_matrix[topic_idx]
            
            # Get top word indices
            top_indices = np.argsort(word_scores)[-num_words:][::-1]
            
            # Create (word, score) pairs
            topic_words = [(self.vocabulary[idx], word_scores[idx]) 
                          for idx in top_indices]
            topics.append(topic_words)
        
        return topics
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to topic space."""
        doc_term_matrix = self.vectorizer.transform(documents)
        return self.svd.transform(doc_term_matrix)


class LDAModel:
    """Latent Dirichlet Allocation model."""
    
    def __init__(self, num_topics: int = 5, alpha: float = 0.1, beta: float = 0.01):
        self.num_topics = num_topics
        self.alpha = alpha  # Document-topic prior
        self.beta = beta    # Topic-word prior
        self.vectorizer = None
        self.lda = None
        self.vocabulary = None
    
    def fit(self, documents: List[str], method: str = 'sklearn'):
        """Fit LDA model to documents."""
        if method == 'sklearn':
            self._fit_sklearn(documents)
        else:
            self._fit_from_scratch(documents)
    
    def _fit_sklearn(self, documents: List[str]):
        """Fit using sklearn's LDA."""
        # Use count vectorizer for LDA
        self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.vocabulary = self.vectorizer.get_feature_names_out()
        
        # Fit LDA
        self.lda = LatentDirichletAllocation(
            n_components=self.num_topics,
            doc_topic_prior=self.alpha,
            topic_word_prior=self.beta,
            random_state=42,
            max_iter=10
        )
        
        self.lda.fit(doc_term_matrix)
    
    def _fit_from_scratch(self, documents: List[str]):
        """Simplified LDA using Gibbs sampling."""
        # Tokenize documents
        tokenized_docs = []
        word_to_id = {}
        id_to_word = {}
        word_id = 0
        
        for doc in documents:
            tokens = doc.lower().split()
            doc_tokens = []
            for token in tokens:
                if token not in word_to_id:
                    word_to_id[token] = word_id
                    id_to_word[word_id] = token
                    word_id += 1
                doc_tokens.append(word_to_id[token])
            tokenized_docs.append(doc_tokens)
        
        self.vocabulary = list(word_to_id.keys())
        vocab_size = len(self.vocabulary)
        
        # Initialize topic assignments randomly
        doc_topic_counts = np.zeros((len(documents), self.num_topics))
        topic_word_counts = np.zeros((self.num_topics, vocab_size))
        topic_counts = np.zeros(self.num_topics)
        
        # Topic assignments for each word
        topic_assignments = []
        
        for doc_idx, doc_tokens in enumerate(tokenized_docs):
            doc_topics = []
            for token in doc_tokens:
                # Random initial assignment
                topic = np.random.randint(self.num_topics)
                doc_topics.append(topic)
                
                # Update counts
                doc_topic_counts[doc_idx, topic] += 1
                topic_word_counts[topic, token] += 1
                topic_counts[topic] += 1
            
            topic_assignments.append(doc_topics)
        
        # Gibbs sampling (simplified - just a few iterations)
        for iteration in range(50):
            for doc_idx, doc_tokens in enumerate(tokenized_docs):
                for word_idx, word_id in enumerate(doc_tokens):
                    # Current topic
                    old_topic = topic_assignments[doc_idx][word_idx]
                    
                    # Remove from counts
                    doc_topic_counts[doc_idx, old_topic] -= 1
                    topic_word_counts[old_topic, word_id] -= 1
                    topic_counts[old_topic] -= 1
                    
                    # Calculate probabilities for each topic
                    probs = np.zeros(self.num_topics)
                    for topic in range(self.num_topics):
                        # P(topic|doc) * P(word|topic)
                        doc_topic_prob = (doc_topic_counts[doc_idx, topic] + self.alpha) / \
                                       (len(doc_tokens) - 1 + self.num_topics * self.alpha)
                        
                        topic_word_prob = (topic_word_counts[topic, word_id] + self.beta) / \
                                        (topic_counts[topic] + vocab_size * self.beta)
                        
                        probs[topic] = doc_topic_prob * topic_word_prob
                    
                    # Sample new topic
                    probs /= probs.sum()
                    new_topic = np.random.choice(self.num_topics, p=probs)
                    
                    # Update assignments and counts
                    topic_assignments[doc_idx][word_idx] = new_topic
                    doc_topic_counts[doc_idx, new_topic] += 1
                    topic_word_counts[new_topic, word_id] += 1
                    topic_counts[new_topic] += 1
        
        # Store final distributions
        self.doc_topic_dist = doc_topic_counts + self.alpha
        self.doc_topic_dist /= self.doc_topic_dist.sum(axis=1, keepdims=True)
        
        self.topic_word_dist = topic_word_counts + self.beta
        self.topic_word_dist /= self.topic_word_dist.sum(axis=1, keepdims=True)
    
    def get_topics(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """Extract top words for each topic."""
        topics = []
        
        if hasattr(self, 'lda') and self.lda is not None:
            # sklearn LDA
            for topic_idx in range(self.num_topics):
                word_scores = self.lda.components_[topic_idx]
                top_indices = np.argsort(word_scores)[-num_words:][::-1]
                
                topic_words = [(self.vocabulary[idx], word_scores[idx]) 
                              for idx in top_indices]
                topics.append(topic_words)
        else:
            # From scratch implementation
            for topic_idx in range(self.num_topics):
                word_scores = self.topic_word_dist[topic_idx]
                top_indices = np.argsort(word_scores)[-num_words:][::-1]
                
                topic_words = [(self.vocabulary[idx], word_scores[idx]) 
                              for idx in top_indices]
                topics.append(topic_words)
        
        return topics
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to topic space."""
        if hasattr(self, 'lda') and self.lda is not None:
            doc_term_matrix = self.vectorizer.transform(documents)
            return self.lda.transform(doc_term_matrix)
        else:
            # Simplified: return uniform distribution
            return np.ones((len(documents), self.num_topics)) / self.num_topics


def perform_lsa(documents: List[str], num_topics: int = 5) -> LSAModel:
    """Perform Latent Semantic Analysis."""
    model = LSAModel(num_topics=num_topics)
    model.fit(documents)
    return model


def perform_lda(documents: List[str], num_topics: int = 5) -> LDAModel:
    """Perform Latent Dirichlet Allocation."""
    model = LDAModel(num_topics=num_topics)
    model.fit(documents)
    return model


def extract_topics(model: Union[LSAModel, LDAModel], 
                  num_words: int = 10) -> List[List[Tuple[str, float]]]:
    """Extract topics from model."""
    return model.get_topics(num_words)


def get_document_topics(model: Union[LSAModel, LDAModel], 
                       document: str) -> List[Tuple[int, float]]:
    """Get topic distribution for a document."""
    doc_topics = model.transform([document])[0]
    
    # Return as (topic_id, probability) pairs
    topic_probs = []
    for topic_idx, prob in enumerate(doc_topics):
        if prob > 0.01:  # Threshold for relevance
            topic_probs.append((topic_idx, prob))
    
    # Sort by probability
    topic_probs.sort(key=lambda x: x[1], reverse=True)
    return topic_probs


def calculate_coherence_score(model: Union[LSAModel, LDAModel], 
                            documents: List[str], 
                            num_words: int = 10) -> float:
    """Calculate topic coherence score (simplified version)."""
    topics = model.get_topics(num_words)
    
    # Simple coherence: average pairwise word co-occurrence
    coherence_scores = []
    
    for topic_words in topics:
        topic_coherence = 0
        word_list = [word for word, _ in topic_words]
        
        # Count co-occurrences
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                word1, word2 = word_list[i], word_list[j]
                co_occur = 0
                occur1 = 0
                
                for doc in documents:
                    doc_lower = doc.lower()
                    if word1 in doc_lower and word2 in doc_lower:
                        co_occur += 1
                    if word1 in doc_lower:
                        occur1 += 1
                
                if occur1 > 0:
                    topic_coherence += co_occur / occur1
        
        # Average coherence for topic
        if len(word_list) > 1:
            topic_coherence /= (len(word_list) * (len(word_list) - 1) / 2)
        
        coherence_scores.append(topic_coherence)
    
    return np.mean(coherence_scores)


def compare_topic_models(documents: List[str], num_topics: int = 5):
    """Compare different topic modeling approaches."""
    results = {}
    
    # LSA
    lsa_model = perform_lsa(documents, num_topics)
    results['LSA'] = {
        'topics': extract_topics(lsa_model, num_words=5),
        'coherence': calculate_coherence_score(lsa_model, documents),
        'explained_variance': lsa_model.svd.explained_variance_ratio_.sum()
    }
    
    # LDA
    lda_model = perform_lda(documents, num_topics)
    results['LDA'] = {
        'topics': extract_topics(lda_model, num_words=5),
        'coherence': calculate_coherence_score(lda_model, documents)
    }
    
    # NMF (Non-negative Matrix Factorization)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    nmf = NMF(n_components=num_topics, random_state=42)
    nmf.fit(doc_term_matrix)
    
    vocabulary = vectorizer.get_feature_names_out()
    nmf_topics = []
    for topic_idx in range(num_topics):
        word_scores = nmf.components_[topic_idx]
        top_indices = np.argsort(word_scores)[-5:][::-1]
        topic_words = [(vocabulary[idx], word_scores[idx]) for idx in top_indices]
        nmf_topics.append(topic_words)
    
    results['NMF'] = {
        'topics': nmf_topics,
        'reconstruction_error': nmf.reconstruction_err_
    }
    
    return results


# Demo functions
if __name__ == "__main__":
    print("Topic Modeling with LSA and LDA\n")
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "Supervised learning requires labeled training data",
        "Unsupervised learning finds patterns without labels",
        "Reinforcement learning learns through trial and error",
        "Transfer learning reuses knowledge from one task to another",
        "Data preprocessing is crucial for machine learning models",
        "Feature engineering improves model performance",
        "Neural networks are inspired by biological neurons",
        "Convolutional neural networks excel at image processing",
        "Recurrent neural networks handle sequential data",
        "Transformers revolutionized natural language processing",
        "BERT and GPT are popular transformer models"
    ]
    
    print(f"Number of documents: {len(documents)}\n")
    
    # Perform LSA
    print("="*50)
    print("Latent Semantic Analysis (LSA)")
    print("="*50)
    
    lsa_model = perform_lsa(documents, num_topics=3)
    lsa_topics = extract_topics(lsa_model, num_words=5)
    
    for i, topic_words in enumerate(lsa_topics):
        print(f"\nTopic {i}:")
        for word, score in topic_words:
            print(f"  {word}: {score:.3f}")
    
    # Document topics
    test_doc = "Deep neural networks for image classification"
    doc_topics = get_document_topics(lsa_model, test_doc)
    print(f"\nDocument topics for '{test_doc}':")
    for topic_id, prob in doc_topics:
        print(f"  Topic {topic_id}: {prob:.3f}")
    
    print("\n" + "="*50)
    print("Latent Dirichlet Allocation (LDA)")
    print("="*50)
    
    lda_model = perform_lda(documents, num_topics=3)
    lda_topics = extract_topics(lda_model, num_words=5)
    
    for i, topic_words in enumerate(lda_topics):
        print(f"\nTopic {i}:")
        for word, score in topic_words:
            print(f"  {word}: {score:.3f}")
    
    # Document topics
    doc_topics = get_document_topics(lda_model, test_doc)
    print(f"\nDocument topics for '{test_doc}':")
    for topic_id, prob in doc_topics:
        print(f"  Topic {topic_id}: {prob:.3f}")
    
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    comparison = compare_topic_models(documents, num_topics=3)
    
    for model_name, results in comparison.items():
        print(f"\n{model_name}:")
        if 'coherence' in results:
            print(f"  Coherence score: {results['coherence']:.3f}")
        if 'explained_variance' in results:
            print(f"  Explained variance: {results['explained_variance']:.3f}")
        if 'reconstruction_error' in results:
            print(f"  Reconstruction error: {results['reconstruction_error']:.3f}")
        
        print("  Topics:")
        for i, topic in enumerate(results['topics']):
            words = [word for word, _ in topic[:3]]
            print(f"    Topic {i}: {', '.join(words)}")
