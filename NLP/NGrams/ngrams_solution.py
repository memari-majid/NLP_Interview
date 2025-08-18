import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import math
import random


class NGramModel:
    """N-gram language model with various smoothing techniques."""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.total_words = 0
        self.unk_token = "<UNK>"
        self.start_token = "<START>"
        self.end_token = "<END>"
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list."""
        # Add start and end tokens
        padded_tokens = [self.start_token] * (n - 1) + tokens + [self.end_token]
        
        ngrams = []
        for i in range(len(padded_tokens) - n + 1):
            ngram = tuple(padded_tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def train(self, texts: List[str]):
        """Train the n-gram model on texts."""
        for text in texts:
            tokens = self.tokenize(text)
            self.vocabulary.update(tokens)
            self.total_words += len(tokens)
            
            # Count n-grams
            ngrams = self.get_ngrams(tokens, self.n)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                
                # Count (n-1)-gram contexts
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1
        
        # Add special tokens to vocabulary
        self.vocabulary.add(self.unk_token)
        self.vocabulary.add(self.start_token)
        self.vocabulary.add(self.end_token)
    
    def get_probability(self, ngram: Tuple[str, ...], smoothing: str = 'none') -> float:
        """Calculate probability of n-gram with optional smoothing."""
        if smoothing == 'none':
            if self.n == 1:
                # Unigram probability
                count = self.ngram_counts.get(ngram, 0)
                return count / self.total_words if self.total_words > 0 else 0
            else:
                # Higher-order n-gram probability
                context = ngram[:-1]
                context_count = self.context_counts.get(context, 0)
                if context_count == 0:
                    return 0
                ngram_count = self.ngram_counts.get(ngram, 0)
                return ngram_count / context_count
        
        elif smoothing == 'laplace':
            # Add-one (Laplace) smoothing
            vocab_size = len(self.vocabulary)
            
            if self.n == 1:
                count = self.ngram_counts.get(ngram, 0)
                return (count + 1) / (self.total_words + vocab_size)
            else:
                context = ngram[:-1]
                context_count = self.context_counts.get(context, 0)
                ngram_count = self.ngram_counts.get(ngram, 0)
                return (ngram_count + 1) / (context_count + vocab_size)
        
        else:
            raise ValueError(f"Unknown smoothing method: {smoothing}")
    
    def calculate_perplexity(self, text: str, smoothing: str = 'laplace') -> float:
        """Calculate perplexity of text under the model."""
        tokens = self.tokenize(text)
        ngrams = self.get_ngrams(tokens, self.n)
        
        if not ngrams:
            return float('inf')
        
        log_prob_sum = 0
        for ngram in ngrams:
            # Handle unknown words
            ngram_with_unk = tuple(
                token if token in self.vocabulary else self.unk_token 
                for token in ngram
            )
            
            prob = self.get_probability(ngram_with_unk, smoothing)
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                # Avoid log(0)
                log_prob_sum += math.log(1e-10)
        
        # Calculate perplexity
        avg_log_prob = log_prob_sum / len(ngrams)
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def generate_next_word(self, context: Tuple[str, ...], 
                          smoothing: str = 'laplace',
                          temperature: float = 1.0) -> str:
        """Generate next word given context."""
        candidates = []
        probabilities = []
        
        # Try all possible next words
        for word in self.vocabulary:
            if word in [self.start_token]:  # Don't generate start token
                continue
            
            ngram = context + (word,)
            prob = self.get_probability(ngram, smoothing)
            
            if prob > 0:
                candidates.append(word)
                # Apply temperature
                if temperature != 1.0:
                    prob = prob ** (1 / temperature)
                probabilities.append(prob)
        
        if not candidates:
            # Fallback to random word
            return random.choice(list(self.vocabulary - {self.start_token}))
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1 / len(candidates) for _ in candidates]
        
        # Sample from distribution
        return np.random.choice(candidates, p=probabilities)
    
    def generate_text(self, seed: str = "", length: int = 50, 
                     smoothing: str = 'laplace',
                     temperature: float = 1.0) -> str:
        """Generate text starting from seed."""
        if seed:
            tokens = self.tokenize(seed)
        else:
            tokens = []
        
        # Initialize context
        if len(tokens) < self.n - 1:
            # Pad with start tokens
            context = [self.start_token] * (self.n - 1 - len(tokens)) + tokens
        else:
            context = tokens[-(self.n - 1):]
        
        generated = tokens.copy()
        
        for _ in range(length):
            next_word = self.generate_next_word(tuple(context), smoothing, temperature)
            
            if next_word == self.end_token:
                break
            
            generated.append(next_word)
            
            # Update context
            if self.n > 1:
                context = context[1:] + [next_word]
            else:
                context = []
        
        return ' '.join(generated)


def build_ngram_model(texts: List[str], n: int = 3) -> NGramModel:
    """Build n-gram language model from texts."""
    model = NGramModel(n=n)
    model.train(texts)
    return model


def calculate_perplexity(model: NGramModel, text: str, 
                        smoothing: str = 'laplace') -> float:
    """Calculate perplexity of text under model."""
    return model.calculate_perplexity(text, smoothing)


def generate_text(model: NGramModel, seed: str = "", length: int = 50,
                 smoothing: str = 'laplace', temperature: float = 1.0) -> str:
    """Generate text from model."""
    return model.generate_text(seed, length, smoothing, temperature)


def apply_smoothing(model: NGramModel, method: str = 'laplace') -> NGramModel:
    """Apply smoothing to model (returns same model with smoothing parameter)."""
    # In this implementation, smoothing is applied during probability calculation
    # This function is here for API compatibility
    return model


class GoodTuringSmoothing:
    """Good-Turing smoothing implementation."""
    
    @staticmethod
    def calculate_frequencies_of_frequencies(ngram_counts: Dict) -> Dict[int, int]:
        """Calculate N_c values (number of n-grams with count c)."""
        freq_of_freq = defaultdict(int)
        for count in ngram_counts.values():
            freq_of_freq[count] += 1
        return dict(freq_of_freq)
    
    @staticmethod
    def smooth_count(count: int, freq_of_freq: Dict[int, int]) -> float:
        """Apply Good-Turing smoothing to a count."""
        if count == 0:
            # Probability for unseen n-grams
            n1 = freq_of_freq.get(1, 0)
            n_total = sum(freq_of_freq.values())
            return n1 / n_total if n_total > 0 else 0
        
        # For seen n-grams
        nc = freq_of_freq.get(count, 0)
        nc_plus_1 = freq_of_freq.get(count + 1, 0)
        
        if nc > 0 and nc_plus_1 > 0:
            return (count + 1) * nc_plus_1 / nc
        else:
            return count  # No smoothing if we can't apply GT


class KneserNeySmoothing:
    """Simplified Kneser-Ney smoothing."""
    
    def __init__(self, discount: float = 0.75):
        self.discount = discount
    
    def calculate_continuation_probability(self, word: str, 
                                         ngram_counts: Dict,
                                         n: int) -> float:
        """Calculate continuation probability for Kneser-Ney."""
        # Count unique contexts where word appears
        continuation_count = 0
        total_continuations = 0
        
        for ngram, count in ngram_counts.items():
            if len(ngram) == n and ngram[-1] == word:
                continuation_count += 1
            if len(ngram) == n:
                total_continuations += 1
        
        return continuation_count / total_continuations if total_continuations > 0 else 0


def compare_smoothing_methods(texts: List[str], test_text: str):
    """Compare different smoothing methods."""
    # Train models with different n values
    results = {}
    
    for n in [1, 2, 3]:
        model = build_ngram_model(texts, n=n)
        
        results[f"{n}-gram"] = {
            'no_smoothing': model.calculate_perplexity(test_text, smoothing='none'),
            'laplace': model.calculate_perplexity(test_text, smoothing='laplace')
        }
    
    return results


def analyze_zipf_law(model: NGramModel) -> Dict[str, int]:
    """Analyze word frequency distribution (Zipf's law)."""
    word_counts = defaultdict(int)
    
    # Count unigrams
    for ngram, count in model.ngram_counts.items():
        if len(ngram) == 1:
            word_counts[ngram[0]] += count
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate Zipf's law statistics
    ranks = list(range(1, len(sorted_words) + 1))
    frequencies = [count for _, count in sorted_words]
    
    # Expected frequencies according to Zipf's law
    if frequencies:
        k = frequencies[0]  # Frequency of most common word
        expected = [k / r for r in ranks]
        
        return {
            'word_frequencies': sorted_words[:10],  # Top 10 words
            'zipf_constant': k,
            'actual_vs_expected': list(zip(frequencies[:10], expected[:10]))
        }
    
    return {}


# Demo functions
if __name__ == "__main__":
    print("N-Gram Language Model Implementation\n")
    
    # Sample training data
    training_texts = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "The cat and the dog are friends",
        "A mat is for a cat",
        "A log is for a dog",
        "Cats like to sit on mats",
        "Dogs like to sit on logs",
        "The quick brown fox jumps over the lazy dog",
        "Natural language processing is fascinating",
        "Machine learning models can process language"
    ]
    
    # Train different n-gram models
    print("Training models...")
    unigram_model = build_ngram_model(training_texts, n=1)
    bigram_model = build_ngram_model(training_texts, n=2)
    trigram_model = build_ngram_model(training_texts, n=3)
    
    print("\n" + "="*50 + "\n")
    
    # Calculate perplexity
    test_texts = [
        "The cat sat on the mat",  # Seen
        "The dog sat on the mat",  # Partially seen
        "The elephant sat on the chair"  # Mostly unseen
    ]
    
    print("Perplexity Analysis:")
    for test_text in test_texts:
        print(f"\nText: '{test_text}'")
        for model, name in [(unigram_model, "Unigram"), 
                          (bigram_model, "Bigram"), 
                          (trigram_model, "Trigram")]:
            perp = calculate_perplexity(model, test_text)
            print(f"  {name} perplexity: {perp:.2f}")
    
    print("\n" + "="*50 + "\n")
    
    # Generate text
    print("Text Generation:")
    seeds = ["The cat", "The dog", ""]
    
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        for model, name in [(bigram_model, "Bigram"), (trigram_model, "Trigram")]:
            generated = generate_text(model, seed, length=20, temperature=0.8)
            print(f"  {name}: {generated}")
    
    print("\n" + "="*50 + "\n")
    
    # Compare smoothing methods
    print("Smoothing Comparison:")
    comparison = compare_smoothing_methods(training_texts, "The cat likes dogs")
    for model_type, results in comparison.items():
        print(f"\n{model_type}:")
        for method, perplexity in results.items():
            print(f"  {method}: {perplexity:.2f}")
    
    print("\n" + "="*50 + "\n")
    
    # Analyze Zipf's law
    print("Zipf's Law Analysis (Bigram model):")
    zipf_analysis = analyze_zipf_law(bigram_model)
    if 'word_frequencies' in zipf_analysis:
        print("\nTop 10 words:")
        for i, (word, count) in enumerate(zipf_analysis['word_frequencies'], 1):
            print(f"  {i}. {word}: {count}")
        
        print("\nZipf's law comparison (actual vs expected):")
        for i, (actual, expected) in enumerate(zipf_analysis['actual_vs_expected'][:5], 1):
            print(f"  Rank {i}: actual={actual}, expected={expected:.1f}")
    
    print("\n" + "="*50 + "\n")
    
    # Character-level n-grams
    print("Character-level trigrams (for morphology):")
    char_model = NGramModel(n=3)
    
    # Train on character level
    char_texts = []
    for text in training_texts[:5]:
        char_texts.append(' '.join(text))  # Space between characters
    
    char_model.train(char_texts)
    
    # Generate character-level text
    char_generated = char_model.generate_text("T h e", length=30)
    print("Generated:", ''.join(char_generated.split()))
