# NLP Interview Questions - What Actually Gets Asked

**Focus on the 15 most common NLP interview questions. Skip the rest.**

## 🎯 TOP 10 MUST-PRACTICE (Asked in 90% of interviews)

| Rank | Question | File | Time | Why Asked |
|------|----------|------|------|-----------|
| 1 | **"Implement TF-IDF from scratch"** | [TFIDF](NLP/TFIDF/) | 30min | Tests fundamental understanding |
| 2 | **"Build text classifier pipeline"** | [Classification](NLP/Text_Classification/) | 25min | Most practical NLP task |
| 3 | **"Handle text tokenization edge cases"** | [Tokenization](NLP/Tokenization/) | 15min | Basic but essential |
| 4 | **"Implement self-attention"** | [Attention](NLP/Attention_Mechanisms/) | 25min | Core of modern NLP |
| 5 | **"Word2Vec training step"** | [Word2Vec](NLP/Embeddings/) | 20min | Classic embeddings |
| 6 | **"Rule-based sentiment analysis"** | [Sentiment](NLP/Sentiment_Analysis/) | 20min | Business-relevant |
| 7 | **"Calculate text similarity"** | [Similarity](NLP/Similarity/) | 25min | Search/recommendation |
| 8 | **"BPE tokenization algorithm"** | [BPE](NLP/Tokenization_Advanced/) | 30min | Modern tokenization |
| 9 | **"BERT fine-tuning setup"** | [BERT](NLP/Transformers/) | 30min | Transfer learning |
| 10 | **"Extract entities with regex"** | [NER](NLP/NER/) | 25min | Information extraction |

## 🏢 Company-Specific Questions

### **OpenAI/Anthropic** (LLM Companies)
**Must practice**: Self-attention, GPT block, text generation
- "Implement the transformer attention mechanism"
- "How would you generate text with different sampling strategies?"
- "Explain instruction tuning vs pre-training"

**Focus**: [Attention](NLP/Attention_Mechanisms/), [GPT Block](NLP/GPT_Implementation/), [Text Generation](NLP/LLM_Fundamentals/)

### **Google/Meta** (Research + Scale)
**Must practice**: BERT fine-tuning, text classification, embeddings
- "Fine-tune BERT for spam detection"
- "Build content moderation classifier"
- "Compare word-level vs subword tokenization"

**Focus**: [BERT](NLP/Transformers/), [Classification](NLP/Text_Classification/), [BPE](NLP/Tokenization_Advanced/)

### **Amazon/Microsoft** (Product-focused)
**Must practice**: Search relevance, sentiment, practical NLP
- "Rank search results by relevance" 
- "Analyze customer review sentiment"
- "Build autocomplete system"

**Focus**: [TF-IDF](NLP/TFIDF/), [Sentiment](NLP/Sentiment_Analysis/), [Text Similarity](NLP/Similarity/)

## ⚡ **1-Week Crash Course** (Interview in 1 week)

**Day 1**: [TF-IDF](NLP/TFIDF/) - Most asked algorithm
**Day 2**: [Text Classification](NLP/Text_Classification/) - Most practical 
**Day 3**: [Tokenization](NLP/Tokenization/) - Always comes up
**Day 4**: [Self-Attention](NLP/Attention_Mechanisms/) - Modern NLP core
**Day 5**: [Word2Vec](NLP/Embeddings/) - Classic but still asked
**Day 6**: [Text Similarity](NLP/Similarity/) - Search/recommendation
**Day 7**: Review all + mock interview

## 🚨 **Common Interview Traps**

### **TF-IDF Questions**
- ❌ "Use sklearn" → ✅ Implement from scratch
- ❌ Skip IDF explanation → ✅ Explain why IDF matters
- ❌ Forget edge cases → ✅ Handle empty documents

### **Attention Questions**  
- ❌ Mention "it's complex" → ✅ Implement step-by-step
- ❌ Skip scaling factor → ✅ Explain why divide by √d_k
- ❌ Forget causal mask → ✅ Handle autoregressive case

### **Tokenization Questions**
- ❌ "Just split on spaces" → ✅ Handle punctuation, contractions
- ❌ Ignore Unicode → ✅ Discuss encoding issues
- ❌ Miss subword benefits → ✅ Explain OOV handling

## 🎪 **Real Interview Examples**

### **Google L4/L5**
*"Implement a function that takes a document collection and a query, returns top-k most relevant documents."*
→ Practice: [TF-IDF](NLP/TFIDF/) + [Text Similarity](NLP/Similarity/)

### **OpenAI/Anthropic**
*"Walk me through how self-attention works and implement the core computation."*
→ Practice: [Self-Attention](NLP/Attention_Mechanisms/)

### **Meta/Facebook**
*"Build a content moderation system that classifies posts as toxic/non-toxic."*
→ Practice: [Text Classification](NLP/Text_Classification/) + [BERT Fine-tuning](NLP/Transformers/)

### **Amazon**
*"How would you analyze customer review sentiment at scale?"*
→ Practice: [Sentiment Analysis](NLP/Sentiment_Analysis/) + scaling discussion

## 📋 **Interview Prep Checklist**

### **Before the Interview**
- [ ] Can implement TF-IDF in 30 minutes
- [ ] Can explain self-attention mechanism clearly
- [ ] Know when to use different tokenization approaches
- [ ] Can build text classification pipeline end-to-end
- [ ] Understand transformer vs RNN trade-offs

### **Day of Interview**
- [ ] Review complexity cheat sheet
- [ ] Practice explaining attention mechanism out loud
- [ ] Rehearse TF-IDF formula explanation
- [ ] Know your edge case handling strategies

## 🎯 **What NOT to Study** (Waste of time for interviews)

❌ **Complex topic modeling** (LDA implementation details)
❌ **Advanced RNN variants** (GRU vs LSTM internals)  
❌ **Speech processing** (rarely asked in NLP roles)
❌ **Extensive regex patterns** (focus on basic entity extraction)
❌ **Deep mathematical proofs** (focus on implementation)

## ⚡ **Quick Practice Commands**

```bash
# Practice most important problems
python NLP/TFIDF/tfidf_solution.py
python NLP/Attention_Mechanisms/self_attention_solution.py
python NLP/Text_Classification/text_classification_solution.py
python NLP/Tokenization/tokenization_solution.py
python NLP/Embeddings/word2vec_solution.py
```

## 📊 **Interview Success Metrics**

### **Minimum Viable Preparation** (1 week)
- ✅ Top 5 problems completed
- ✅ Can explain transformer attention
- ✅ Know TF-IDF formula by heart
- ✅ Understand tokenization trade-offs

### **Strong Preparation** (2 weeks)  
- ✅ Top 10 problems mastered
- ✅ Can implement any on whiteboard
- ✅ Know scaling considerations
- ✅ Ready for follow-up questions

### **Exceptional Preparation** (1 month)
- ✅ All 26 problems completed
- ✅ Can teach concepts to others
- ✅ Know cutting-edge developments
- ✅ Ready for principal/staff level interviews

---

**Focus on what matters. Master the top 10. Skip the rest unless you have extra time.** 🎯

**Expected interview success rate**: 90%+ with top 10 mastery
