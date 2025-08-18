# Anki-Friendly Code Design Guidelines

This guide ensures all solutions are optimized for Anki memorization with bite-sized, atomic cards.

## 📏 Card Size Guidelines

### ✅ **Optimal Card Sizes**
- **Code snippets**: 5-15 lines max
- **Formulas**: 1-2 lines
- **Concepts**: 1-3 bullet points  
- **Review time**: 10-30 seconds per card

### ❌ **Too Large**
- Functions over 20 lines
- Multiple concepts in one card
- Long explanations
- Review time over 1 minute

## 🎯 Solution Structure for Anki

### 1. **Break Functions into Atomic Pieces**

Instead of:
```python
def tfidf_vectorizer(documents):
    # 50+ lines of code...
```

Write:
```python
def build_vocabulary(documents):
    """Build vocabulary from documents."""
    # 5-10 lines
    
def calculate_tf(doc, vocab):
    """Calculate term frequency."""
    # 5-10 lines
    
def calculate_idf(documents, vocab):
    """Calculate inverse document frequency."""
    # 5-10 lines
```

### 2. **Add Anki-Friendly Comments**

```python
# FORMULA: IDF = log((N+1)/(df+1)) + 1
# KEY: Add 1 to avoid division by zero
# COMPLEXITY: O(n*m) where n=docs, m=vocab
# EDGE: Empty documents return zero vector
```

### 3. **One Concept Per Function**

Good for Anki:
```python
def tokenize_basic(text):
    """Basic tokenization - just split on whitespace."""
    return text.lower().split()

def tokenize_punctuation(text):
    """Handle punctuation separately."""
    import re
    return re.findall(r'\w+|[^\w\s]', text.lower())
```

Bad for Anki:
```python
def tokenize_advanced(text, method='basic', handle_punctuation=True, 
                     remove_stopwords=False, stem=True):
    # Too many concepts in one function!
```

## 🃏 Card Types Generated

### 1. **Problem Understanding** (What's the approach?)
- Front: Problem title + "What's the key insight?"
- Back: Brief problem description + hint

### 2. **Implementation** (Code the solution)
- Front: "Implement function_name()"  
- Back: 5-15 lines of code

### 3. **Formula** (Remember the math)
- Front: "Write the TF-IDF formula"
- Back: `TF × log(N/df)`

### 4. **Complexity** (Big-O analysis)
- Front: "What's the complexity?"
- Back: "Time: O(n), Space: O(1)"

### 5. **Edge Cases** (Handle errors)
- Front: "What edge cases to handle?"
- Back: Empty input, None values, etc.

### 6. **Key Insights** (Interview tips)
- Front: "Interview talking points?"
- Back: Main algorithm choice, trade-offs

## 📝 Writing Anki-Optimized Solutions

### Template Structure

```python
"""
Problem: [Name]
Anki Cards: 5-7 atomic functions
"""

# Card 1: Main algorithm insight
def algorithm_choice():
    """
    KEY: Use TF-IDF for document ranking
    WHY: Balances term frequency with document rarity
    """
    pass

# Card 2: Core formula
def compute_score(tf, idf):
    """
    FORMULA: score = tf * idf
    """
    return tf * idf

# Card 3: Key helper function
def calculate_tf(term_count, total_terms):
    """Calculate term frequency."""
    # EDGE: Handle zero total_terms
    if total_terms == 0:
        return 0
    return term_count / total_terms

# Card 4: Another helper
def calculate_idf(num_docs, docs_with_term):
    """
    FORMULA: IDF = log((N+1)/(df+1)) + 1
    """
    import math
    return math.log((num_docs + 1) / (docs_with_term + 1)) + 1

# Card 5: Edge case handling  
def handle_empty_document(doc):
    """EDGE: Empty documents get zero vector."""
    if not doc or not doc.strip():
        return {}
    return process_document(doc)

# Card 6: Complexity note
"""
COMPLEXITY:
- Time: O(n*m) where n=documents, m=vocabulary
- Space: O(n*m) for TF-IDF matrix
- OPTIMIZE: Use sparse matrices for large vocab
"""
```

## 🎪 Mobile Optimization

### Code Formatting for Small Screens

```python
# Good: Short lines, clear breaks
def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = sum(x**2 for x in a)**0.5
    norm_b = sum(x**2 for x in b)**0.5
    return dot / (norm_a * norm_b)

# Bad: Long lines that need scrolling
def cosine_similarity(vector_a, vector_b): return sum(a*b for a,b in zip(vector_a, vector_b)) / (sum(a**2 for a in vector_a)**0.5 * sum(b**2 for b in vector_b)**0.5)
```

### Variable Names
- Use short but clear names: `doc` not `document_text`
- Common abbreviations: `tf`, `idf`, `vocab`, `sim`

## 📊 Metrics for Good Anki Cards

### Review Statistics Target
- **Again rate**: < 20% (cards aren't too hard)
- **Hard rate**: < 30% (appropriate difficulty)
- **Good rate**: 40-60% (well-sized cards)
- **Easy rate**: < 20% (cards aren't too simple)

### Card Creation Guidelines
- **Functions per solution**: 3-6 atomic functions
- **Lines per function**: 5-15 lines
- **Total cards per problem**: 5-8 cards
- **Review time**: 10-30 seconds per card

## 🔄 Converting Existing Solutions

### Before (Too Large)
```python
def implement_tfidf(documents):
    # 100+ lines doing everything
    # Hard to memorize in chunks
```

### After (Anki-Friendly)
```python
# Split into 5-6 cards:
build_vocabulary()      # Card 1: 10 lines
compute_tf()           # Card 2: 8 lines  
compute_idf()          # Card 3: 8 lines
vectorize_document()   # Card 4: 12 lines
cosine_similarity()    # Card 5: 10 lines
# Total: Same functionality, better for memory
```

## 💡 Quick Checklist

Before committing a solution, ensure:
- [ ] No function exceeds 20 lines
- [ ] Each function has one clear purpose
- [ ] Key formulas are commented
- [ ] Complexity is noted
- [ ] Edge cases are marked
- [ ] Total creates 5-8 Anki cards
- [ ] Mobile-friendly formatting

## 🚀 Testing Your Cards

Run the optimized converter:
```bash
python convert_to_anki_optimized.py
```

Import and review:
- Do cards load quickly?
- Can you review in 10-30 seconds?
- Is code readable on mobile?
- Are concepts atomic?

Remember: **Better to have 6 simple cards than 1 complex card!**
