# 🎯 Interview Quick Reference

**Last-minute prep guide for NLP coding interviews**

## 🔥 TOP 5 MUST-KNOW (Practice these if interview is tomorrow)

### 1. **TF-IDF Implementation** ([Solution](NLP/TFIDF/tfidf_solution.py))
**What they ask**: "Implement TF-IDF from scratch and find most similar documents"
**Key points to mention**:
- TF = term_freq / doc_length (normalized frequency)
- IDF = log(total_docs / docs_containing_term) (rarity measure)
- Cosine similarity measures angle, not magnitude
- Time complexity: O(d×v), Space: O(d×v)

### 2. **Self-Attention** ([Solution](NLP/Attention_Mechanisms/self_attention_solution.py))
**What they ask**: "Explain and implement the attention mechanism"
**Key points to mention**:
- Q, K, V matrices from linear projections
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Scaling by √d_k prevents gradient vanishing
- Causal masking for autoregressive models

### 3. **Text Classification** ([Solution](NLP/Text_Classification/text_classification_solution.py))
**What they ask**: "Build a spam/sentiment classifier end-to-end"
**Key points to mention**:
- Feature extraction is most important step
- TF-IDF + Logistic Regression is solid baseline
- Handle class imbalance with weights
- Evaluate with precision/recall, not just accuracy

### 4. **Word2Vec Skip-gram** ([Solution](NLP/Embeddings/word2vec_solution.py))
**What they ask**: "How does Word2Vec work? Implement training step"
**Key points to mention**:
- Skip-gram: predict context from center word
- Co-occurring words get similar embeddings
- Dot product → sigmoid → gradient descent
- Negative sampling needed for efficiency

### 5. **Tokenization** ([Solution](NLP/Tokenization/tokenization_solution.py))
**What they ask**: "Implement tokenizer handling contractions and punctuation"
**Key points to mention**:
- Regex approach most robust: `\w+(?:'\w+)?|[^\w\s]`
- Handle contractions as single tokens
- Always test edge cases (empty, None, Unicode)
- Subword tokenization (BPE) better for rare words

## ⚡ Quick Concepts to Explain

### **When asked "What is..."**
- **TF-IDF**: Balances term frequency with document rarity
- **Attention**: Mechanism to focus on relevant parts of input
- **BERT vs GPT**: Bidirectional vs autoregressive training
- **BPE**: Subword tokenization using most frequent character pairs
- **Perplexity**: How surprised model is by text (lower = better)

### **When asked "How would you optimize..."**
- **TF-IDF for large corpora**: Sparse matrices, inverted indices
- **Attention for long sequences**: Linear attention, sliding windows
- **Word embeddings**: Negative sampling, hierarchical softmax
- **Text classification**: Feature selection, ensemble methods

### **When asked "Compare X vs Y"**
- **Rule-based vs ML sentiment**: Speed vs accuracy trade-off
- **Word-level vs subword tokenization**: OOV handling
- **CNN vs LSTM for text**: Parallel vs sequential processing
- **Static vs contextual embeddings**: Word2Vec vs BERT

## 🚨 Common Gotchas

### **Implementation Traps**
- ❌ `log(0)` in TF-IDF → ✅ Add small epsilon or handle separately
- ❌ Overflow in sigmoid/softmax → ✅ Subtract max before exp()
- ❌ Forget to normalize cosine similarity → ✅ Divide by norms
- ❌ Inconsistent vocabulary between train/test → ✅ Fit on train only

### **Conceptual Traps**
- ❌ "Attention is just weighted sum" → ✅ Explain learned relevance
- ❌ "BERT is just bidirectional GPT" → ✅ Different training objectives
- ❌ "More data always better" → ✅ Quality > quantity, bias concerns

## 📝 Explanation Templates

### **For Any Algorithm**
1. **Problem it solves**: "This algorithm addresses..."
2. **High-level approach**: "The key insight is..."
3. **Step-by-step breakdown**: "First we..., then we..., finally..."
4. **Why it works**: "This works because..."
5. **Limitations**: "However, it struggles with..."
6. **Alternatives**: "Other approaches include..."

### **For Complexity Analysis**
- **Time**: "For each document/word/token, we do..."
- **Space**: "We need to store..."
- **Bottlenecks**: "The expensive part is..."
- **Scaling**: "For larger inputs, we could..."

## 🎪 Last-Minute Practice

### **If interview is in 1 hour**
1. Run through TF-IDF solution explaining each step out loud
2. Draw attention mechanism on paper
3. Review the 5 "Key points to mention" above

### **If interview is tomorrow**
1. Implement TF-IDF from scratch without looking
2. Explain self-attention mechanism to a friend
3. Code text classification pipeline end-to-end
4. Review company-specific focus areas

### **If interview is next week**
Practice all top 10 problems from main README

---

**Remember**: Explain your thinking process out loud. Interviewers care more about your reasoning than perfect code.

**Good luck!** 🚀
