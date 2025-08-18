import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from gensim.models import Word2Vec, KeyedVectors
    from gensim.models.callbacks import CallbackAny2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Install gensim: pip install gensim")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class Word2VecFromScratch:
    """Simplified Word2Vec implementation for educational purposes."""
    
    def __init__(self, embedding_dim: int = 100, window_size: int = 5, 
                 learning_rate: float = 0.01, epochs: int = 5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocabulary = {}
        self.word_embeddings = None
        self.context_embeddings = None
    
    def build_vocabulary(self, sentences: List[List[str]], min_count: int = 2):
        """Build vocabulary from sentences."""
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        # Filter by minimum count
        self.vocabulary = {word: idx for idx, (word, count) in 
                          enumerate(word_counts.items()) if count >= min_count}
        self.vocab_size = len(self.vocabulary)
        
        # Initialize embeddings randomly
        self.word_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        self.context_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def train_skipgram(self, sentences: List[List[str]]):
        """Train Skip-gram model."""
        for epoch in range(self.epochs):
            total_loss = 0
            pairs = 0
            
            for sentence in sentences:
                for center_pos, center_word in enumerate(sentence):
                    if center_word not in self.vocabulary:
                        continue
                    
                    center_idx = self.vocabulary[center_word]
                    
                    # Get context words
                    for context_pos in range(max(0, center_pos - self.window_size),
                                           min(len(sentence), center_pos + self.window_size + 1)):
                        if context_pos == center_pos:
                            continue
                        
                        context_word = sentence[context_pos]
                        if context_word not in self.vocabulary:
                            continue
                        
                        context_idx = self.vocabulary[context_word]
                        
                        # Positive sample
                        center_vec = self.word_embeddings[center_idx]
                        context_vec = self.context_embeddings[context_idx]
                        
                        # Forward pass
                        score = np.dot(center_vec, context_vec)
                        prediction = self.sigmoid(score)
                        
                        # Loss (binary cross-entropy)
                        loss = -np.log(prediction + 1e-10)
                        total_loss += loss
                        
                        # Backward pass (gradient descent)
                        grad = (prediction - 1)
                        self.word_embeddings[center_idx] -= self.learning_rate * grad * context_vec
                        self.context_embeddings[context_idx] -= self.learning_rate * grad * center_vec
                        
                        pairs += 1
            
            if pairs > 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss/pairs:.4f}")
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector."""
        if word in self.vocabulary:
            return self.word_embeddings[self.vocabulary[word]]
        return None
    
    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words."""
        if word not in self.vocabulary:
            return []
        
        word_vec = self.get_vector(word)
        similarities = []
        
        for other_word, idx in self.vocabulary.items():
            if other_word == word:
                continue
            
            other_vec = self.word_embeddings[idx]
            
            # Cosine similarity
            similarity = np.dot(word_vec, other_vec) / (
                np.linalg.norm(word_vec) * np.linalg.norm(other_vec) + 1e-10
            )
            similarities.append((other_word, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def train_word2vec(sentences: List[List[str]], embedding_dim: int = 100,
                  architecture: str = 'skip-gram', min_count: int = 2) -> Word2Vec:
    """Train Word2Vec model using gensim."""
    if not GENSIM_AVAILABLE:
        print("Using simplified implementation")
        model = Word2VecFromScratch(embedding_dim=embedding_dim)
        model.build_vocabulary(sentences, min_count=min_count)
        model.train_skipgram(sentences)
        return model
    
    # Gensim parameters
    sg = 1 if architecture == 'skip-gram' else 0  # 1 for skip-gram, 0 for CBOW
    
    # Train model
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=min_count,
        workers=4,
        sg=sg,
        epochs=10,
        seed=42
    )
    
    return model


def solve_analogy(model, word_a: str, word_b: str, word_c: str) -> str:
    """Solve word analogy: a is to b as c is to ?"""
    if not GENSIM_AVAILABLE:
        if isinstance(model, Word2VecFromScratch):
            # Simple implementation
            vec_a = model.get_vector(word_a)
            vec_b = model.get_vector(word_b)
            vec_c = model.get_vector(word_c)
            
            if any(v is None for v in [vec_a, vec_b, vec_c]):
                return "OOV"
            
            # Calculate target vector: b - a + c
            target_vec = vec_b - vec_a + vec_c
            
            # Find most similar to target
            best_word = None
            best_similarity = -1
            
            for word, idx in model.vocabulary.items():
                if word in [word_a, word_b, word_c]:
                    continue
                
                vec = model.word_embeddings[idx]
                similarity = np.dot(target_vec, vec) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_word = word
            
            return best_word if best_word else "NOT_FOUND"
    
    try:
        # Use gensim's built-in method
        result = model.wv.most_similar(
            positive=[word_b, word_c],
            negative=[word_a],
            topn=1
        )
        return result[0][0] if result else "NOT_FOUND"
    except KeyError:
        return "OOV"


def find_similar_words(model, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Find most similar words to given word."""
    if isinstance(model, Word2VecFromScratch):
        return model.most_similar(word, top_k)
    
    try:
        return model.wv.most_similar(word, topn=top_k)
    except KeyError:
        return []


def visualize_embeddings(model, words: List[str], method: str = 'tsne'):
    """Visualize word embeddings in 2D."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available")
        return
    
    # Get vectors
    vectors = []
    valid_words = []
    
    for word in words:
        try:
            if isinstance(model, Word2VecFromScratch):
                vec = model.get_vector(word)
                if vec is not None:
                    vectors.append(vec)
                    valid_words.append(word)
            else:
                vec = model.wv[word]
                vectors.append(vec)
                valid_words.append(word)
        except KeyError:
            continue
    
    if len(vectors) < 2:
        print("Not enough valid words for visualization")
        return
    
    vectors = np.array(vectors)
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2)
    
    vectors_2d = reducer.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
    
    for i, word in enumerate(valid_words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title(f'Word Embeddings Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_embeddings(sentences: List[List[str]], 
                      methods: List[str] = ['word2vec', 'glove', 'fasttext']) -> Dict:
    """Compare different embedding methods (simplified demo)."""
    results = {}
    
    # Word2Vec
    if 'word2vec' in methods:
        if GENSIM_AVAILABLE:
            w2v_model = train_word2vec(sentences, embedding_dim=50)
            results['word2vec'] = {
                'vocabulary_size': len(w2v_model.wv),
                'embedding_dim': w2v_model.wv.vector_size,
                'model': w2v_model
            }
    
    # Note: GloVe and fastText would require additional implementations
    # or loading pre-trained models
    
    return results


def handle_oov_words(model, word: str, subword_size: int = 3) -> np.ndarray:
    """Handle out-of-vocabulary words using character n-grams."""
    if not isinstance(model, Word2VecFromScratch):
        try:
            return model.wv[word]
        except KeyError:
            pass
    
    # Character n-gram approach (simplified)
    vectors = []
    
    # Get character n-grams
    for i in range(len(word) - subword_size + 1):
        subword = word[i:i + subword_size]
        
        # Check if any word in vocabulary contains this subword
        for vocab_word, idx in (model.vocabulary.items() if isinstance(model, Word2VecFromScratch) 
                               else model.wv.key_to_index.items()):
            if subword in vocab_word:
                if isinstance(model, Word2VecFromScratch):
                    vectors.append(model.word_embeddings[idx])
                else:
                    vectors.append(model.wv[vocab_word])
                break
    
    if vectors:
        # Average the vectors
        return np.mean(vectors, axis=0)
    else:
        # Random vector as last resort
        dim = model.embedding_dim if isinstance(model, Word2VecFromScratch) else model.wv.vector_size
        return np.random.randn(dim) * 0.1


def evaluate_analogies(model, analogy_file: str = None) -> float:
    """Evaluate model on word analogy task."""
    # Common analogies for testing
    test_analogies = [
        ("king", "man", "queen", "woman"),
        ("paris", "france", "berlin", "germany"),
        ("good", "better", "bad", "worse"),
        ("walk", "walked", "run", "ran"),
        ("cat", "kitten", "dog", "puppy")
    ]
    
    correct = 0
    total = 0
    
    for a, b, c, expected in test_analogies:
        result = solve_analogy(model, a, b, c)
        if result == expected:
            correct += 1
        total += 1
        print(f"{a}:{b} :: {c}:{result} (expected: {expected})")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


# Demo data generator
def create_sample_corpus() -> List[List[str]]:
    """Create sample corpus for demonstration."""
    corpus = [
        "the king ruled the kingdom with wisdom".split(),
        "the queen was beloved by her people".split(),
        "the prince and princess lived in the castle".split(),
        "the man worked hard every day".split(),
        "the woman was very intelligent and kind".split(),
        "paris is the capital of france".split(),
        "berlin is the capital of germany".split(),
        "london is the capital of england".split(),
        "cats and dogs are popular pets".split(),
        "kittens and puppies are very cute".split(),
        "python is a programming language".split(),
        "java is also a programming language".split(),
        "machine learning requires mathematics".split(),
        "deep learning uses neural networks".split(),
        "natural language processing is fascinating".split()
    ]
    return corpus


if __name__ == "__main__":
    print("Word2Vec Training and Applications\n")
    
    # Create sample corpus
    sentences = create_sample_corpus()
    print(f"Training on {len(sentences)} sentences...\n")
    
    # Train Word2Vec model
    print("Training Word2Vec (Skip-gram)...")
    model = train_word2vec(sentences, embedding_dim=50, architecture='skip-gram')
    
    print("\n" + "="*50 + "\n")
    
    # Test word similarities
    test_words = ["king", "python", "learning"]
    for word in test_words:
        print(f"\nMost similar to '{word}':")
        similar = find_similar_words(model, word, top_k=3)
        for similar_word, score in similar:
            print(f"  {similar_word}: {score:.3f}")
    
    print("\n" + "="*50 + "\n")
    
    # Test analogies
    print("Word Analogies:")
    analogies = [
        ("king", "man", "queen"),
        ("paris", "france", "berlin"),
        ("python", "programming", "french"),
        ("cat", "kitten", "dog")
    ]
    
    for a, b, c in analogies:
        result = solve_analogy(model, a, b, c)
        print(f"{a} - {b} + {c} = {result}")
    
    print("\n" + "="*50 + "\n")
    
    # Handle OOV words
    print("Out-of-Vocabulary Handling:")
    oov_words = ["javascript", "kingship", "queenly"]
    for word in oov_words:
        vec = handle_oov_words(model, word)
        print(f"{word}: vector shape = {vec.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # Evaluate on analogies
    print("Analogy Evaluation:")
    accuracy = evaluate_analogies(model)
    print(f"\nAnalogy accuracy: {accuracy:.2%}")
    
    # Visualize embeddings (if available)
    if VISUALIZATION_AVAILABLE and not isinstance(model, Word2VecFromScratch):
        print("\nGenerating embedding visualization...")
        words_to_plot = ["king", "queen", "man", "woman", "paris", "france", 
                        "berlin", "germany", "python", "java", "learning"]
        visualize_embeddings(model, words_to_plot, method='pca')
