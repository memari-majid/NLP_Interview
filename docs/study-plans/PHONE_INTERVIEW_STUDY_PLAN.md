# Phone Interview Memorization Plan

Purpose: memorize how to explain and implement the most asked NLP problems for phone screens (15–30 minutes). Uses active recall, spaced repetition, and short verbal scripts.

## Core Answer Template (Use on every call)
1) Restate the problem and constraints (10s)
2) Outline approach and why it fits (20–30s)
3) Complexity (time/space) and trade-offs (10–20s)
4) Edge cases + example dry-run (30–45s)
5) Code highlights (data shapes, loops, key formula) (30–60s)
6) Validate + alternatives (15–30s)

## Active Recall Drills (Daily)
- One-minute script: explain an algorithm from memory (no notes)
- 5 bullet recall: formula, shapes, loop, mask/edge, complexity
- Whiteboard from memory: outline function signature and core loop
- Dry-run one tiny example end-to-end

## Spaced Repetition Schedule (Top 10)
- Today → +1 day → +3 days → +7 days → +14 days
- Each review: 2-minute explanation + 1 dry-run + 1 follow-up question

Topics (Top 10):
- TF-IDF: [Problem](NLP/TFIDF/tfidf_problem.md) · [Solution](NLP/TFIDF/tfidf_solution.py)
- Text Classification: [Problem](NLP/Text_Classification/text_classification_problem.md) · [Solution](NLP/Text_Classification/text_classification_solution.py)
- Tokenization: [Problem](NLP/Tokenization/tokenization_problem.md) · [Solution](NLP/Tokenization/tokenization_solution.py)
- Self-Attention: [Problem](NLP/Attention_Mechanisms/self_attention_problem.md) · [Solution](NLP/Attention_Mechanisms/self_attention_solution.py)
- Word2Vec: [Problem](NLP/Embeddings/word2vec_problem.md) · [Solution](NLP/Embeddings/word2vec_solution.py)
- Sentiment (VADER): [Problem](NLP/Sentiment_Analysis/vader_sentiment_problem.md) · [Solution](NLP/Sentiment_Analysis/vader_sentiment_solution.py)
- Similarity: [Problem](NLP/Similarity/cosine_similarity_problem.md) · [Solution](NLP/Similarity/cosine_similarity_solution.py)
- BPE: [Problem](NLP/Tokenization_Advanced/bpe_tokenization_problem.md) · [Solution](NLP/Tokenization_Advanced/bpe_tokenization_solution.py)
- BERT Fine-tuning: [Problem](NLP/Transformers/bert_sentiment_problem.md) · [Solution](NLP/Transformers/bert_sentiment_solution.py)
- Regex NER: [Problem](NLP/Regex_NLP/regex_patterns_problem.md) · [Solution](NLP/Regex_NLP/regex_patterns_solution.py)

## 7-Day Phone Screen Sprint
- Day 1: TF-IDF + Similarity → Recite IDF formula; compute a tiny example
- Day 2: Text Classification → Pipeline steps; train/predict path
- Day 3: Tokenization + Stop Words → Edge cases; Unicode; contractions
- Day 4: Self-Attention → Scaled dot-product; masking; shapes
- Day 5: Word2Vec → Negative sampling step; gradients intuition
- Day 6: BPE + Regex NER → Merge loop; pattern boundaries
- Day 7: Mock call (2 problems timed) + review weak spots

Daily routine (20–30 min):
- 5 min: flash recall (scripts) for 2 topics
- 10 min: code outline from memory for 1 topic
- 10 min: dry-run + follow-up questions

## 14-Day Memorization Plan (All Topics)
- D1: TF-IDF · Similarity
- D2: Classification · Stop Words
- D3: Tokenization · Stemming/Lemmatization
- D4: POS Tagging · NER (Regex/statistical)
- D5: BoW · N-grams
- D6: Word2Vec · CNN for Text
- D7: Self-Attention · BERT
- D8: GPT Block · Text Generation
- D9: BPE · Utilities (Normalization)
- D10: Sentiment (VADER) · Sequence Models (LSTM)
- D11: Neural Fundamentals (Perceptron) · Topic Modeling (LSA/LDA)
- D12: Model Evaluation (LLMs) · Fine-Tuning
- D13: Instruction Tuning · Mixed Review
- D14: 2× Mock calls

## One‑Minute Scripts (Top 5)
- TF-IDF: token counts → TF; corpus rarity → IDF = ln((N+1)/(df+1)) + 1; weight = TF×IDF; cosine similarity for ranking; handle empty docs; O(V) per doc
- Self-Attention: scores = QKᵀ/√d; mask; softmax; weighted V; multi-head for subspaces; shapes [B,H,T,d]; causal mask for autoregressive
- Text Classification: preprocess → vectorize → train (LR/linear) → predict; metrics (precision/recall); class imbalance; leakage
- Word2Vec: objective maximizes dot(target, context); negative sampling with sigmoid; update target/context vectors; subsampling frequent words
- BPE: start as chars; merge most frequent pair; repeat K times; improves OOV handling; trade-offs: sequence length vs vocab size

## Phone Call Checklists
- Be explicit: mention shapes, complexity, edge cases
- Narrate while coding: what and why
- Verify with a tiny example before/after coding
- If blocked: propose fallback (e.g., degrade to BoW), discuss trade-offs

## Progress Tracker (Top 10)
- [ ] TF-IDF
- [ ] Text Classification
- [ ] Tokenization
- [ ] Self-Attention
- [ ] Word2Vec
- [ ] Sentiment (VADER)
- [ ] Similarity
- [ ] BPE
- [ ] BERT Fine-tuning
- [ ] Regex NER

Tip: Pair this plan with INTERVIEW_QUICK_REFERENCE.md for last‑minute refresh.

