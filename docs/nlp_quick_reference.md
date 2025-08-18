# NLP Quick Reference Guide

## Essential Imports
```python
# Core libraries
import spacy
import nltk
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, LdaModel

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
```

## Common Patterns

### Text Preprocessing Pipeline
```python
def preprocess_text(text):
    # 1. Lower case
    text = text.lower()
    
    # 2. Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 3. Tokenize
    doc = nlp(text)
    
    # 4. Remove stop words
    tokens = [token.text for token in doc if not token.is_stop]
    
    # 5. Lemmatize
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    
    return ' '.join(tokens)
```

### Quick Text Classification
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Train
text_clf.fit(X_train, y_train)

# Predict
predictions = text_clf.predict(X_test)
```

### Semantic Similarity
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
embeddings = model.encode(['sentence 1', 'sentence 2'])

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
```

### Zero-shot Classification
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification")

result = classifier(
    "This is a tutorial about Transformers",
    candidate_labels=["education", "politics", "business"]
)
```

## Performance Tips

1. **Batch Processing**: Always process multiple texts together
2. **Model Caching**: Load models once, reuse for multiple predictions
3. **GPU Acceleration**: Use `device=0` for GPU when available
4. **Preprocessing**: Cache preprocessed texts when possible
5. **Vector Storage**: Use FAISS or Annoy for similarity search at scale

## Common Gotchas

- **Tokenization**: Different models use different tokenizers (don't mix!)
- **Stop Words**: Be careful removing negations ('not', 'no')
- **Case Sensitivity**: Some models are case-sensitive (cased BERT)
- **Max Length**: Transformer models have max sequence length (usually 512)
- **Memory**: Large models need significant RAM/VRAM
