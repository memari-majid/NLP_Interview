# 🎯 NLP Interview Master Index

## Complete Resource Map for NLP Interview Preparation

### 📖 Learning Paths

#### Path 1: Theory First (Recommended for Beginners)
1. Start with [Theory Flashcards](data/nlp_theory_flashcards.json) - 60 concepts
2. Review [Comprehensive Study Guide](docs/comprehensive_nlp_study_guide.md) - 6 week plan
3. Practice with [Coding Problems](NLP/) - 27 implementations
4. Test with [Company-Specific Questions](docs/interview-guides/COMPANY_SPECIFIC_GUIDE.md)

#### Path 2: Practice First (For Experienced)
1. Jump to [Problem Finder](scripts/problem_finder.py) - Interactive navigation
2. Use [Solution Patterns](docs/interview-guides/SOLUTION_PATTERNS.md) - 10 templates
3. Review gaps with [Theory Flashcards](data/nlp_theory_flashcards.json)
4. Polish with [Quick Reference](docs/nlp_quick_reference.md)

#### Path 3: Interview in 1 Week
1. Focus on [Top 10 Problems](#top-problems)
2. Memorize [Key Formulas](#key-formulas)
3. Review [Company Guide](docs/interview-guides/COMPANY_SPECIFIC_GUIDE.md) for your target
4. Practice with [Practical Flashcards](data/nlp_practical_flashcards.json)

---

## 📚 Complete Topic Coverage

### Fundamentals
- **Tokenization**: [Problem](NLP/Tokenization/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_001)
- **Stemming/Lemmatization**: [Problem](NLP/Stemming_Lemmatization/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_002)
- **Stop Words**: [Problem](NLP/Stop_Word_Removal/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_019)
- **N-grams**: [Problem](NLP/NGrams/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_032)
- **Text Normalization**: [Problem](NLP/Utilities/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_031)

### Embeddings & Representations
- **Bag of Words**: [Problem](NLP/BoW_Vectors/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_059)
- **TF-IDF**: [Problem](NLP/TFIDF/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_018)
- **Word2Vec**: [Problem](NLP/Embeddings/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_003)
- **GloVe**: Theory Card](data/nlp_theory_flashcards.json#nlp_005)
- **Contextualized Embeddings**: [Theory Card](data/nlp_theory_flashcards.json#nlp_026)

### Classical ML for NLP
- **Text Classification**: [Problem](NLP/Text_Classification/) | [Practical Card](data/nlp_practical_flashcards.json#practical_010)
- **Naive Bayes**: [Problem](NLP/Text_Classification/)
- **SVM for Text**: [Practical Card](data/nlp_practical_flashcards.json#practical_010)
- **Similarity Metrics**: [Problem](NLP/Similarity/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_044)

### Deep Learning
- **RNN/LSTM**: [Problem](NLP/Sequence_Models/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_055)
- **CNN for Text**: [Problem](NLP/CNN_Text/)
- **Seq2Seq Models**: [Theory Card](data/nlp_theory_flashcards.json#nlp_055)

### Transformers & Attention
- **Self-Attention**: [Problem](NLP/Attention_Mechanisms/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_006)
- **Multi-Head Attention**: [Theory Card](data/nlp_theory_flashcards.json#nlp_008)
- **BERT**: [Problem](NLP/Transformers/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_010)
- **GPT**: [Problem](NLP/GPT_Implementation/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_010)
- **T5/BART**: [Theory Card](data/nlp_theory_flashcards.json#nlp_011)
- **Position Encodings**: [Theory Card](data/nlp_theory_flashcards.json#nlp_009)

### NLP Applications
- **Named Entity Recognition**: [Problem](NLP/NER/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_020)
- **POS Tagging**: [Problem](NLP/POS_Tagging/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_021)
- **Sentiment Analysis**: [Problem](NLP/Sentiment_Analysis/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_035)
- **Question Answering**: [Theory Card](data/nlp_theory_flashcards.json#nlp_023)
- **Text Summarization**: [Theory Card](data/nlp_theory_flashcards.json#nlp_051)
- **Machine Translation**: [Theory Card](data/nlp_theory_flashcards.json#nlp_034)
- **Topic Modeling**: [Problem](NLP/TopicModeling/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_033)

### Large Language Models
- **Fine-tuning**: [Problem](NLP/Fine_Tuning/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_012)
- **Instruction Tuning**: [Problem](NLP/Instruction_Tuning/) | [Theory Card](data/nlp_theory_flashcards.json#nlp_014)
- **RLHF**: [Theory Card](data/nlp_theory_flashcards.json#nlp_014)
- **In-Context Learning**: [Theory Card](data/nlp_theory_flashcards.json#nlp_013)
- **Chain-of-Thought**: [Theory Card](data/nlp_theory_flashcards.json#nlp_027)
- **RAG Systems**: [Theory Card](data/nlp_theory_flashcards.json#nlp_038)
- **Prompt Engineering**: [Theory Card](data/nlp_theory_flashcards.json#nlp_012)

### Evaluation & Metrics
- **Perplexity**: [Theory Card](data/nlp_theory_flashcards.json#nlp_017)
- **BLEU Score**: [Theory Card](data/nlp_theory_flashcards.json#nlp_016)
- **ROUGE Score**: [Theory Card](data/nlp_theory_flashcards.json#nlp_049)
- **F1 Score**: [Theory Card](data/nlp_theory_flashcards.json#nlp_040)
- **BERTScore**: [Theory Card](data/nlp_theory_flashcards.json#nlp_016)

### Production & Optimization
- **Quantization**: [Theory Card](data/nlp_theory_flashcards.json#nlp_054)
- **Distillation**: [Practical Card](data/nlp_practical_flashcards.json#advanced_003)
- **PEFT Methods**: [Theory Card](data/nlp_theory_flashcards.json#nlp_050)
- **Deployment**: [Practical Card](data/nlp_practical_flashcards.json#advanced_003)

---

## 🔥 Top Problems

### Must-Do Before Any Interview
1. [TF-IDF Implementation](NLP/TFIDF/)
2. [Self-Attention Mechanism](NLP/Attention_Mechanisms/)
3. [Text Classification Pipeline](NLP/Text_Classification/)
4. [Named Entity Recognition](NLP/NER/)
5. [Word2Vec Concepts](NLP/Embeddings/)

### By Company Focus
- **OpenAI/Anthropic**: GPT Implementation, Instruction Tuning, RLHF Theory
- **Google**: TF-IDF, Attention, Large-scale considerations
- **Meta**: Applied ML, Classification, Social media text
- **Microsoft**: End-to-end systems, Azure integration
- **Amazon**: Customer focus, production deployment

---

## 📝 Key Formulas

### Essential Math
```
TF-IDF = TF(t,d) × log(N/DF(t))
Attention(Q,K,V) = softmax(QK^T/√d_k)V
Cosine Similarity = (A·B) / (||A|| × ||B||)
F1 = 2 × (precision × recall) / (precision + recall)
Perplexity = exp(-1/N Σ log P(w_i|context))
```

---

## 🛠️ Tools & Scripts

### Interactive Tools
```bash
# Find problems by difficulty/company/topic
python scripts/problem_finder.py

# Generate Anki flashcards
python scripts/convert_theory_to_anki.py      # Theory cards
python scripts/convert_to_anki_optimized.py    # Coding cards

# Extract from cheatsheet
python scripts/extract_cheatsheet_content.py

# Build knowledge base
python scripts/create_comprehensive_nlp_kb.py
```

### Quick Commands
```bash
# Run any solution
python NLP/TFIDF/tfidf_solution.py

# Search for a topic
grep -r "attention" NLP/

# Find problems by difficulty
find NLP -name "*_problem.md" | xargs grep -l "Difficulty: Medium"
```

---

## 📊 Coverage Statistics

### Content Metrics
- **Coding Problems**: 27 with solutions
- **Theory Flashcards**: 60 concepts
- **Practical Flashcards**: 21 implementations
- **Code Snippets**: 30+ ready-to-use
- **Formulas**: 6 essential
- **Libraries Covered**: 6 major
- **Models Explained**: 5 architectures
- **Datasets Documented**: 6 common

### Learning Materials
- **Study Guides**: 3 comprehensive
- **Company Guides**: 10+ companies
- **Solution Patterns**: 10 templates
- **Quick References**: 4 documents
- **Anki Decks**: 3 types

---

## 🎓 Study Recommendations

### Week 1: Foundations
- Master preprocessing pipeline
- Understand embeddings (static vs contextual)
- Practice TF-IDF and BoW

### Week 2: Classical ML
- Text classification with sklearn
- Similarity metrics
- Feature engineering

### Week 3: Deep Learning
- RNN/LSTM for sequences
- Attention mechanism
- Encoder-decoder architecture

### Week 4: Transformers
- BERT vs GPT
- Fine-tuning strategies
- Position encodings

### Week 5: LLMs
- Prompting techniques
- RLHF and alignment
- Scaling laws

### Week 6: Production
- Optimization techniques
- Deployment strategies
- System design

---

## 📈 Success Metrics

Track your progress:
- [ ] Can implement TF-IDF from scratch
- [ ] Understand attention mechanism deeply
- [ ] Know when to use BERT vs GPT
- [ ] Can design end-to-end NLP system
- [ ] Familiar with 5+ evaluation metrics
- [ ] Can optimize models for production
- [ ] Know company-specific focus areas

---

## 🔗 External Resources

### Documentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [spaCy Documentation](https://spacy.io/api)
- [NLTK Book](https://www.nltk.org/book/)

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)

### Courses
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)
- [fast.ai NLP Course](https://www.fast.ai/posts/2019-07-08-fastai-nlp.html)

---

## 💡 Final Tips

1. **Start with fundamentals** - Don't jump to transformers without understanding basics
2. **Code everything** - Implementation deepens understanding
3. **Use flashcards daily** - Spaced repetition works
4. **Practice explaining** - Teaching solidifies knowledge
5. **Focus on trade-offs** - Interviewers love discussing pros/cons
6. **Know your target company** - Tailor preparation to their focus
7. **Build something** - A project demonstrates practical skills

Good luck with your NLP interviews! 🚀