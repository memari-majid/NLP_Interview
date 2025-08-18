# NLP Coding Interview Practice

This repository is a comprehensive bank of NLP coding interview questions with production-ready Python solutions. No courses, just practical problems and answers you can run immediately.

## Quick Start
1. Pick a topic from the list below
2. Read the problem statement (`*_problem.md`)
3. Try solving it yourself
4. Check the solution (`*_solution.py`)
5. Run the solution to see it in action

Each solution file is self-contained with example usage and handles missing dependencies gracefully [[memory:4362663]].

## Complete Problem Bank

### 1. Text Preprocessing

#### Tokenization
- **Problem**: `NLP/Tokenization/tokenization_problem.md`
- **Solution**: `NLP/Tokenization/tokenization_solution.py`
- **Topics**: Word tokenization, punctuation handling, contractions

#### Stop Word Removal
- **Problem**: `NLP/Stop_Word_Removal/stopword_removal_problem.md`
- **Solution**: `NLP/Stop_Word_Removal/stopword_removal_solution.py`
- **Topics**: Configurable stopwords, case handling, language support

#### Stemming & Lemmatization
- **Problem**: `NLP/Stemming_Lemmatization/stemming_lemmatization_problem.md`
- **Solution**: `NLP/Stemming_Lemmatization/stemming_lemmatization_solution.py`
- **Topics**: Porter stemmer, WordNet lemmatizer, POS-aware processing

### 2. Language Analysis

#### Part-of-Speech (POS) Tagging
- **Problem**: `NLP/POS_Tagging/pos_tagging_problem.md`
- **Solution**: `NLP/POS_Tagging/pos_tagging_solution.py`
- **Topics**: Penn Treebank tags, noun phrase extraction, ambiguity handling

#### Named Entity Recognition (NER)
- **Problem**: `NLP/NER/ner_problem.md`
- **Solution**: `NLP/NER/ner_solution.py`
- **Topics**: spaCy NER, custom entities (email/phone), entity relationships

### 3. Feature Extraction

#### TF-IDF Implementation
- **Problem**: `NLP/TFIDF/tfidf_problem.md`
- **Solution**: `NLP/TFIDF/tfidf_solution.py`
- **Topics**: From-scratch implementation, document similarity, BM25 comparison

#### Text Similarity
- **Problem**: `NLP/Similarity/cosine_similarity_problem.md`
- **Solution**: `NLP/Similarity/cosine_similarity_solution.py`
- **Topics**: Cosine/Jaccard similarity, semantic similarity, MinHash LSH

### 4. Machine Learning

#### Text Classification
- **Problem**: `NLP/Text_Classification/text_classification_problem.md`
- **Solution**: `NLP/Text_Classification/text_classification_solution.py`
- **Topics**: TF-IDF + LogisticRegression, multi-label classification, feature importance

#### LSTM Sentiment Analysis
- **Problem**: `NLP/Sequence_Models/lstm_sentiment_problem.md`
- **Solution**: `NLP/Sequence_Models/lstm_sentiment_solution.py`
- **Topics**: PyTorch/TensorFlow LSTM, attention mechanism, GRU comparison

#### BERT Fine-tuning
- **Problem**: `NLP/Transformers/bert_sentiment_problem.md`
- **Solution**: `NLP/Transformers/bert_sentiment_solution.py`
- **Topics**: Hugging Face transformers, fine-tuning strategies, attention visualization

### 5. Utilities

#### Text Normalization
- **Problem**: `NLP/Utilities/text_normalization_problem.md`
- **Solution**: `NLP/Utilities/text_normalization_solution.py`
- **Topics**: URL/email handling, contraction expansion, Unicode normalization

## Dependencies

Solutions use standard Python NLP libraries:
- **Core**: numpy, scikit-learn
- **NLP**: nltk, spacy
- **Deep Learning**: torch/tensorflow, transformers
- **Optional**: gensim, sentence-transformers

Each solution includes fallbacks when libraries are missing.

## Interview Tips

1. **Start Simple**: Implement basic version first, then optimize
2. **Handle Edge Cases**: Empty strings, single words, special characters
3. **Discuss Trade-offs**: Speed vs accuracy, memory vs computation
4. **Know Complexity**: Time and space complexity for each algorithm
5. **Production Considerations**: Scaling, error handling, monitoring

## Common Interview Patterns

### String Manipulation
- Tokenization, regex patterns
- Unicode and encoding issues

### Statistical Methods
- TF-IDF, n-grams, co-occurrence
- Similarity metrics, clustering

### Machine Learning
- Feature engineering for text
- Model selection and evaluation
- Handling imbalanced data

### Deep Learning
- Embeddings and representations
- Sequence modeling (RNN/LSTM)
- Transfer learning (BERT/GPT)

### System Design
- Real-time vs batch processing
- Scalability and caching
- Model serving and updates

## Quick Reference

### Complexity Cheat Sheet
- Tokenization: O(n) where n = text length
- TF-IDF: O(n*m) where n = docs, m = vocab size
- LSTM: O(t*h²) where t = sequence length, h = hidden size
- BERT: O(n²) where n = sequence length (self-attention)

### When to Use What
- **Bag-of-Words**: Simple classification, interpretability needed
- **TF-IDF**: Document retrieval, keyword extraction
- **Word2Vec**: Semantic similarity, analogies
- **LSTM**: Sequential patterns, sentiment analysis
- **BERT**: State-of-the-art accuracy, limited training data

## Contributing

Feel free to add more problems! Follow the existing pattern:
1. Create `topic_problem.md` with clear problem statement
2. Create `topic_solution.py` with runnable solution
3. Include example usage in `if __name__ == "__main__"`
4. Handle missing dependencies gracefully

---

**Remember**: In interviews, explain your approach before coding. Good luck! 🚀