# NLP Solution Patterns & Templates

Common patterns and templates for solving NLP interview problems efficiently.

## 🎯 Pattern 1: Text Preprocessing Pipeline

### When to Use
- Any problem starting with raw text
- When asked about "cleaning" or "normalization"

### Template
```python
def preprocess_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters (keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 3. Normalize whitespace
    text = ' '.join(text.split())
    
    # 4. Handle edge cases
    if not text or text.isspace():
        return ""
    
    return text

def tokenize(text: str) -> List[str]:
    # 1. Basic split
    tokens = text.split()
    
    # 2. Or regex for better handling
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    return tokens
```

### Common Variations
- Unicode normalization: `unicodedata.normalize('NFKD', text)`
- Preserve punctuation: `re.findall(r'\w+|[^\w\s]', text)`
- Subword tokenization: Use BPE or WordPiece

## 🎯 Pattern 2: Vectorization (Sparse)

### When to Use
- Converting text to numerical features
- TF-IDF, Bag of Words implementations

### Template
```python
def build_vocabulary(documents: List[str]) -> Dict[str, int]:
    vocab = {}
    for doc in documents:
        tokens = tokenize(doc)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def vectorize_document(doc: str, vocab: Dict[str, int]) -> List[float]:
    vector = [0.0] * len(vocab)
    tokens = tokenize(doc)
    
    # Count frequencies
    token_counts = Counter(tokens)
    
    # Fill vector
    for token, count in token_counts.items():
        if token in vocab:
            vector[vocab[token]] = count
            
    return vector
```

### Optimization Tips
- Use `defaultdict(int)` for counting
- Consider sparse representations for large vocabularies
- Pre-compute IDF values for TF-IDF

## 🎯 Pattern 3: Sliding Window (N-grams, CNN)

### When to Use  
- N-gram extraction
- CNN convolution
- Local pattern detection

### Template
```python
def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def sliding_window_operation(sequence: List[float], 
                           window_size: int,
                           operation: Callable) -> List[float]:
    results = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        result = operation(window)
        results.append(result)
    return results
```

### Edge Cases
- Handle sequences shorter than window size
- Padding strategies: 'valid', 'same', 'causal'

## 🎯 Pattern 4: Dynamic Programming (Edit Distance, Alignment)

### When to Use
- String similarity problems
- Sequence alignment
- Optimization problems

### Template
```python
def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    
    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n]
```

### Space Optimization
```python
# Use only 2 rows instead of full matrix
def edit_distance_optimized(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    
    prev = list(range(len(s2) + 1))
    curr = [0] * (len(s2) + 1)
    
    for i in range(1, len(s1) + 1):
        curr[0] = i
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev
    
    return prev[len(s2)]
```

## 🎯 Pattern 5: Attention Mechanism

### When to Use
- Transformer components
- Sequence-to-sequence with focus
- Modern NLP architectures

### Template
```python
def scaled_dot_product_attention(Q: np.ndarray, 
                                K: np.ndarray, 
                                V: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> np.ndarray:
    # 1. Compute attention scores
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # 2. Apply mask if provided
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # 3. Apply softmax
    attention_weights = softmax(scores, axis=-1)
    
    # 4. Apply attention to values
    output = np.matmul(attention_weights, V)
    
    return output

def create_causal_mask(size: int) -> np.ndarray:
    # Lower triangular matrix
    mask = np.triu(np.ones((size, size)), k=1)
    return mask
```

### Multi-Head Extension
```python
def multi_head_attention(x: np.ndarray, 
                        num_heads: int, 
                        d_model: int) -> np.ndarray:
    d_k = d_model // num_heads
    
    # Split into heads
    # ... (reshape and transpose)
    
    # Apply attention per head
    # ... (loop or vectorized)
    
    # Concatenate heads
    # ... (reshape back)
    
    return output
```

## 🎯 Pattern 6: Embedding Operations

### When to Use
- Word2Vec, GloVe implementations  
- Similarity computations
- Semantic search

### Template
```python
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    # Compute dot product
    dot_product = np.dot(v1, v2)
    
    # Compute norms
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle zero vectors
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    # Compute cosine similarity
    return dot_product / (norm_v1 * norm_v2)

def nearest_neighbors(query: np.ndarray, 
                     embeddings: np.ndarray, 
                     k: int = 5) -> List[int]:
    # Compute similarities
    similarities = []
    for i, embedding in enumerate(embeddings):
        sim = cosine_similarity(query, embedding)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k indices
    return [idx for idx, _ in similarities[:k]]
```

### Batch Operations
```python
def batch_cosine_similarity(queries: np.ndarray, 
                           keys: np.ndarray) -> np.ndarray:
    # Normalize
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    keys_norm = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    
    # Compute all similarities at once
    similarities = np.matmul(queries_norm, keys_norm.T)
    
    return similarities
```

## 🎯 Pattern 7: Text Generation

### When to Use
- Language model decoding
- Sequence generation
- Sampling strategies

### Template
```python
def generate_text(model, 
                 prompt: str, 
                 max_length: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9) -> str:
    tokens = tokenize(prompt)
    
    for _ in range(max_length):
        # Get model predictions
        logits = model.forward(tokens)
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = softmax(logits[-1])
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_probs, top_k_indices = top_k_filtering(probs, top_k)
            probs = top_k_probs
            indices = top_k_indices
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            probs, indices = nucleus_filtering(probs, indices, top_p)
        
        # Sample next token
        next_token_idx = np.random.choice(indices, p=probs)
        tokens.append(next_token_idx)
        
        # Check for end token
        if next_token_idx == END_TOKEN:
            break
    
    return detokenize(tokens)
```

## 🎯 Pattern 8: Classification Pipeline

### When to Use
- Text classification tasks
- Sentiment analysis
- Any supervised NLP task

### Template
```python
class TextClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def preprocess(self, texts: List[str]) -> List[str]:
        return [preprocess_text(text) for text in texts]
    
    def fit(self, texts: List[str], labels: List[int]):
        # 1. Preprocess
        processed_texts = self.preprocess(texts)
        
        # 2. Vectorize
        self.vectorizer = TfidfVectorizer()  # or custom
        X = self.vectorizer.fit_transform(processed_texts)
        
        # 3. Train model
        self.model = LogisticRegression()  # or any classifier
        self.model.fit(X, labels)
        
    def predict(self, texts: List[str]) -> List[int]:
        # 1. Preprocess
        processed_texts = self.preprocess(texts)
        
        # 2. Vectorize
        X = self.vectorizer.transform(processed_texts)
        
        # 3. Predict
        return self.model.predict(X)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        predictions = self.predict(texts)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted')
        }
```

## 🎯 Pattern 9: Efficient Search

### When to Use
- Document retrieval
- Similarity search
- Question answering

### Template
```python
class EfficientSearcher:
    def __init__(self):
        self.documents = []
        self.index = None
        
    def build_index(self, documents: List[str]):
        self.documents = documents
        
        # Build inverted index
        self.inverted_index = defaultdict(set)
        for doc_id, doc in enumerate(documents):
            tokens = tokenize(doc)
            for token in tokens:
                self.inverted_index[token].add(doc_id)
        
        # Pre-compute TF-IDF or embeddings
        self.doc_vectors = self._compute_vectors(documents)
        
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        # 1. Get candidate documents (inverted index)
        query_tokens = tokenize(query)
        candidates = set()
        for token in query_tokens:
            candidates.update(self.inverted_index.get(token, set()))
        
        # 2. Rank candidates
        query_vector = self._compute_vector(query)
        scores = []
        for doc_id in candidates:
            score = cosine_similarity(query_vector, self.doc_vectors[doc_id])
            scores.append((doc_id, score))
        
        # 3. Return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
```

## 🎯 Pattern 10: Streaming/Online Processing

### When to Use
- Real-time NLP applications
- Large-scale processing
- Memory-constrained environments

### Template
```python
class StreamingProcessor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.statistics = defaultdict(int)
        
    def process_stream(self, text_stream):
        for text in text_stream:
            # Process individual item
            result = self.process_item(text)
            
            # Update buffer
            self.buffer.append(result)
            
            # Update rolling statistics
            self.update_statistics(result)
            
            # Emit results if needed
            if self.should_emit():
                yield self.emit_results()
    
    def process_item(self, text: str):
        # Lightweight processing
        tokens = tokenize(text)
        return {
            'tokens': tokens,
            'length': len(tokens),
            'timestamp': time.time()
        }
    
    def update_statistics(self, result):
        # Update counts, averages, etc.
        for token in result['tokens']:
            self.statistics[token] += 1
```

## 💡 Common Edge Cases to Handle

### Empty Input
```python
if not text or text.isspace():
    return []  # or appropriate empty result
```

### Unicode/Encoding
```python
try:
    text = text.encode('utf-8').decode('utf-8')
except UnicodeDecodeError:
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
```

### Numerical Stability
```python
# Log-sum-exp trick for softmax
def stable_softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)
```

### Memory Efficiency
```python
# Generator instead of list
def process_large_corpus(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield preprocess_text(line.strip())
```

## 🚀 Performance Optimization Tips

1. **Vectorize Operations**
   ```python
   # Bad: Loop
   similarities = []
   for vec in vectors:
       similarities.append(cosine_similarity(query, vec))
   
   # Good: Vectorized
   similarities = np.dot(vectors, query) / (norms * query_norm)
   ```

2. **Use Appropriate Data Structures**
   - `set` for membership testing
   - `defaultdict` for counting
   - `deque` for queues
   - `heapq` for top-k operations

3. **Precompute When Possible**
   - IDF values in TF-IDF
   - Vocabulary mappings
   - Normalized vectors

4. **Early Stopping**
   ```python
   # In search/similarity tasks
   if score < threshold:
       continue  # Skip expensive computations
   ```

## 📋 Interview Communication Templates

### Explaining Approach
"I'll solve this in three steps:
1. [Preprocessing] - Clean and tokenize the text
2. [Core Algorithm] - Apply the main technique
3. [Post-processing] - Format and return results"

### Discussing Complexity
"Time Complexity: O(n*m) where n is [documents] and m is [vocabulary]
Space Complexity: O(m) for storing [the vocabulary mapping]
We could optimize this by [using sparse representations]"

### Handling Edge Cases
"I'm considering these edge cases:
- Empty input: Return empty result
- Unicode text: Normalize using NFKD
- Very long documents: Use streaming approach"

### Trade-off Discussion
"We have a trade-off between [accuracy] and [speed]:
- Option A: More accurate but O(n²) time
- Option B: Approximate but O(n log n) time
Given the requirements, I'd choose [B] because [reason]"
