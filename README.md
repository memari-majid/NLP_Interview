# NLP Coding Interview Question Bank

**19 interview-ready problems with concise Python solutions. Each problem designed for 15-30 minute coding sessions.**

## Quick Practice Format

1. Read problem statement (`*_problem.md`)
2. Implement solution yourself (15-30 min)
3. Compare with provided solution (`*_solution.py`)
4. Run solution to verify understanding

## Problem Bank

### Text Preprocessing (4 problems)
- **[Tokenization](NLP/Tokenization/)** (15 min) - Handle contractions, punctuation
- **[Stop Words](NLP/Stop_Word_Removal/)** (15 min) - Filter common words
- **[Stemming/Lemmatization](NLP/Stemming_Lemmatization/)** (20 min) - Word normalization
- **[Text Normalization](NLP/Utilities/)** (20 min) - Unicode, cleaning

### Language Analysis (3 problems)
- **[POS Tagging](NLP/POS_Tagging/)** (20 min) - Grammar analysis
- **[Named Entity Recognition](NLP/NER/)** (25 min) - Entity extraction
- **[Regex Patterns](NLP/Regex_NLP/)** (20 min) - Pattern matching

### Vector Representations (4 problems)
- **[TF-IDF](NLP/TFIDF/)** (30 min) - Document similarity from scratch
- **[Bag of Words](NLP/BoW_Vectors/)** (20 min) - Vocabulary building
- **[Word2Vec](NLP/Embeddings/)** (20 min) - Skip-gram algorithm
- **[Text Similarity](NLP/Similarity/)** (25 min) - Distance metrics

### Language Modeling (2 problems)
- **[N-gram Models](NLP/NGrams/)** (25 min) - Bigram probabilities
- **[Topic Modeling](NLP/TopicModeling/)** (30 min) - LSA/LDA basics

### Sentiment Analysis (2 problems)
- **[Rule-based VADER](NLP/Sentiment_Analysis/)** (20 min) - Lexicon approach
- **[ML Classification](NLP/Text_Classification/)** (25 min) - Feature engineering

### Neural Networks (4 problems)
- **[Neural Net from Scratch](NLP/Neural_Fundamentals/)** (30 min) - Backpropagation
- **[CNN for Text](NLP/CNN_Text/)** (25 min) - 1D convolution
- **[LSTM](NLP/Sequence_Models/)** (25 min) - Gate mechanisms
- **[BERT Fine-tuning](NLP/Transformers/)** (30 min) - Transfer learning

## Interview Tips

**Before coding:**
- Clarify requirements and edge cases
- Discuss algorithm choice and trade-offs
- Start with simplest approach

**During coding:**
- Think out loud
- Test with examples as you go
- Handle empty/invalid inputs

**Key concepts to explain:**
- Time/space complexity
- Why this approach over alternatives
- Production considerations (scaling, edge cases)

## Complexity Quick Reference

| Algorithm | Time | Space |
|-----------|------|-------|
| Tokenization | O(n) | O(n) |
| TF-IDF | O(d×v) | O(d×v) |
| Word2Vec | O(T×K) | O(V×K) |
| LSTM | O(t×h²) | O(h) |

*n=text length, d=docs, v=vocab, T=training steps, K=embedding dim, t=sequence length, h=hidden size*

## Usage

Each solution is self-contained and runs independently:

```bash
python NLP/Tokenization/tokenization_solution.py
python NLP/TFIDF/tfidf_solution.py
python NLP/Embeddings/word2vec_solution.py
```

**Dependencies**: Most solutions use only Python stdlib + numpy for algorithmic focus.

---

**Total practice time**: ~8 hours to complete all problems  
**Interview prep**: Practice 1-2 problems daily for 2 weeks
