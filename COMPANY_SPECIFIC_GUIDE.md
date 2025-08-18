# Company-Specific NLP Interview Guide

Detailed breakdown of what each company asks and how to prepare efficiently.

## 🚀 OpenAI / Anthropic (LLM Focus)

### What They Look For
- Deep understanding of transformer architecture
- Practical experience with LLMs
- Optimization and efficiency mindset
- Safety and alignment considerations

### Must-Practice Problems
1. **Self-Attention Implementation** [🔴 Hard]
   - [Problem](NLP/Attention_Mechanisms/self_attention_problem.md) | [Solution](NLP/Attention_Mechanisms/self_attention_solution.py)
   - Follow-up: Multi-head attention, positional encoding
   - Time: 30 min

2. **GPT Block from Scratch** [🔴 Hard]
   - [Problem](NLP/GPT_Implementation/gpt_block_problem.md) | [Solution](NLP/GPT_Implementation/gpt_block_solution.py)
   - Follow-up: Layer normalization placement, GELU vs ReLU
   - Time: 35 min

3. **BPE Tokenization** [🔴 Hard]
   - [Problem](NLP/Tokenization_Advanced/bpe_tokenization_problem.md) | [Solution](NLP/Tokenization_Advanced/bpe_tokenization_solution.py)
   - Follow-up: Handling Unicode, vocabulary size trade-offs
   - Time: 30 min

4. **Text Generation Strategies** [🔴 Hard]
   - [Problem](NLP/LLM_Fundamentals/text_generation_problem.md) | [Solution](NLP/LLM_Fundamentals/text_generation_solution.py)
   - Follow-up: Temperature scaling, top-k vs top-p
   - Time: 25 min

### Common Interview Flow
1. Warm-up: Tokenization concepts (10 min)
2. Main: Implement core transformer component (30 min)
3. Discussion: Scaling, optimization, safety (20 min)

### Preparation Strategy
- Week 1: Master attention mechanism
- Week 2: GPT architecture and generation
- Week 3: Tokenization and evaluation

## 🔍 Google (Search & Scale)

### What They Look For
- Scalability mindset
- Classical + modern NLP knowledge
- System design thinking
- Efficiency analysis

### Must-Practice Problems
1. **TF-IDF for Document Ranking** [🟡 Medium]
   - [Problem](NLP/TFIDF/tfidf_problem.md) | [Solution](NLP/TFIDF/tfidf_solution.py)
   - Follow-up: Distributed computation, sparse matrices
   - Time: 25 min

2. **BERT Fine-tuning** [🔴 Hard]
   - [Problem](NLP/Transformers/bert_sentiment_problem.md) | [Solution](NLP/Transformers/bert_sentiment_solution.py)
   - Follow-up: Efficient fine-tuning, distillation
   - Time: 30 min

3. **Efficient Tokenization** [🟢 Easy]
   - [Problem](NLP/Tokenization/tokenization_problem.md) | [Solution](NLP/Tokenization/tokenization_solution.py)
   - Follow-up: Multilingual handling, streaming
   - Time: 15 min

4. **Word2Vec at Scale** [🟡 Medium]
   - [Problem](NLP/Embeddings/word2vec_problem.md) | [Solution](NLP/Embeddings/word2vec_solution.py)
   - Follow-up: Negative sampling, hierarchical softmax
   - Time: 25 min

### Common Interview Flow
1. Algorithm implementation (25 min)
2. Scale discussion (15 min)
3. System design component (20 min)

### Preparation Strategy
- Focus on both classical (TF-IDF) and modern (BERT)
- Practice complexity analysis
- Think distributed systems

## 👥 Meta (Social & Applied ML)

### What They Look For
- Practical ML applications
- Content understanding
- Real-time processing
- Multimodal thinking

### Must-Practice Problems
1. **Text Classification Pipeline** [🟡 Medium]
   - [Problem](NLP/Text_Classification/text_classification_problem.md) | [Solution](NLP/Text_Classification/text_classification_solution.py)
   - Follow-up: Online learning, class imbalance
   - Time: 25 min

2. **CNN for Text** [🔴 Hard]
   - [Problem](NLP/CNN_Text/cnn_text_classification_problem.md) | [Solution](NLP/CNN_Text/cnn_text_classification_solution.py)
   - Follow-up: Filter sizes, character-level CNN
   - Time: 30 min

3. **Real-time Sentiment** [🟡 Medium]
   - [Problem](NLP/Sentiment_Analysis/vader_sentiment_problem.md) | [Solution](NLP/Sentiment_Analysis/vader_sentiment_solution.py)
   - Follow-up: Streaming updates, multilingual
   - Time: 20 min

4. **Similarity at Scale** [🟢 Easy]
   - [Problem](NLP/Similarity/cosine_similarity_problem.md) | [Solution](NLP/Similarity/cosine_similarity_solution.py)
   - Follow-up: Approximate methods, LSH
   - Time: 20 min

### Common Interview Flow
1. Practical problem (20 min)
2. Implementation (25 min)
3. Production considerations (15 min)

### Preparation Strategy
- Focus on applied problems
- Consider real-time constraints
- Think about user experience

## 📦 Amazon (Product & Customer Focus)

### What They Look For
- Customer obsession in solutions
- Practical, working code
- Search and recommendation
- Operational excellence

### Must-Practice Problems
1. **Search Relevance (TF-IDF)** [🟡 Medium]
   - [Problem](NLP/TFIDF/tfidf_problem.md) | [Solution](NLP/TFIDF/tfidf_solution.py)
   - Follow-up: Query expansion, personalization
   - Time: 25 min

2. **Review Sentiment Analysis** [🟡 Medium]
   - [Problem](NLP/Sentiment_Analysis/vader_sentiment_problem.md) | [Solution](NLP/Sentiment_Analysis/vader_sentiment_solution.py)
   - Follow-up: Aspect-based sentiment, multilingual
   - Time: 20 min

3. **Product NER** [🟡 Medium]
   - [Problem](NLP/NER/ner_problem.md) | [Solution](NLP/NER/ner_solution.py)
   - Follow-up: Custom entities, product matching
   - Time: 25 min

4. **Query Understanding** [🟢 Easy]
   - [Problem](NLP/Stop_Word_Removal/stopword_removal_problem.md) | [Solution](NLP/Stop_Word_Removal/stopword_removal_solution.py)
   - Follow-up: Query intent, spell correction
   - Time: 15 min

### Common Interview Flow
1. Customer problem discussion (10 min)
2. Implementation (30 min)
3. Metrics and testing (20 min)

### Preparation Strategy
- Think customer-first
- Focus on search/recommendation
- Consider A/B testing

## 🪟 Microsoft (Enterprise & Integration)

### What They Look For
- Enterprise-scale thinking
- API design skills
- Cross-platform considerations
- Azure integration knowledge

### Must-Practice Problems
1. **Text Classification API** [🟡 Medium]
   - [Problem](NLP/Text_Classification/text_classification_problem.md) | [Solution](NLP/Text_Classification/text_classification_solution.py)
   - Follow-up: Batch processing, model versioning
   - Time: 25 min

2. **BERT for Business** [🔴 Hard]
   - [Problem](NLP/Transformers/bert_sentiment_problem.md) | [Solution](NLP/Transformers/bert_sentiment_solution.py)
   - Follow-up: Model serving, edge deployment
   - Time: 30 min

3. **Document Processing** [🟢 Easy]
   - [Problem](NLP/Utilities/text_normalization_problem.md) | [Solution](NLP/Utilities/text_normalization_solution.py)
   - Follow-up: Office formats, PDF extraction
   - Time: 20 min

### Common Interview Flow
1. Problem understanding (15 min)
2. Design and implementation (30 min)
3. Integration discussion (15 min)

## 🍎 Apple (Privacy & On-Device)

### What They Look For
- Privacy-first solutions
- On-device efficiency
- Mobile optimization
- User experience focus

### Must-Practice Problems
1. **Efficient Embeddings** [🟡 Medium]
   - Focus: Compressed Word2Vec, quantization
   - Time: 25 min

2. **On-device Classification** [🟡 Medium]
   - Focus: Model size, inference speed
   - Time: 25 min

3. **Private Text Processing** [🟢 Easy]
   - Focus: Differential privacy, federated learning
   - Time: 20 min

## 📊 Interview Format Comparison

| Company | Coding | System Design | ML Theory | Behavioral |
|---------|--------|---------------|-----------|------------|
| OpenAI | 60% | 20% | 20% | Low |
| Google | 50% | 30% | 20% | Medium |
| Meta | 60% | 20% | 20% | High |
| Amazon | 50% | 20% | 10% | Very High |
| Microsoft | 40% | 40% | 20% | Medium |
| Apple | 50% | 30% | 20% | Medium |

## 🎯 Efficient Preparation Timeline

### 2 Weeks Before
- Identify target companies
- Complete company-specific must-practice problems
- Review company engineering blogs

### 1 Week Before
- Mock interviews with company-specific problems
- Review common follow-ups
- Practice explaining trade-offs

### Day Before
- Review company values/principles
- Quick review of implemented solutions
- Prepare questions about the company

## 💡 Company-Specific Tips

### OpenAI/Anthropic
- Discuss safety and alignment
- Show knowledge of recent papers
- Emphasize efficiency at scale

### Google
- Think MapReduce for everything
- Discuss distributed systems
- Know complexity by heart

### Meta
- User impact stories
- Real-time considerations
- Content moderation angles

### Amazon
- Customer obsession examples
- Operational metrics
- Frugality in solutions

### Microsoft
- Enterprise integration
- Cross-platform thinking
- Developer experience

## 🔗 Quick Links by Company

### OpenAI Path
[Attention](NLP/Attention_Mechanisms/) → [GPT](NLP/GPT_Implementation/) → [BPE](NLP/Tokenization_Advanced/) → [Generation](NLP/LLM_Fundamentals/)

### Google Path
[TF-IDF](NLP/TFIDF/) → [Word2Vec](NLP/Embeddings/) → [BERT](NLP/Transformers/) → [Scale Discussion]

### Meta Path
[Classification](NLP/Text_Classification/) → [CNN](NLP/CNN_Text/) → [Sentiment](NLP/Sentiment_Analysis/) → [Production]

### Amazon Path
[Search](NLP/TFIDF/) → [Sentiment](NLP/Sentiment_Analysis/) → [NER](NLP/NER/) → [Customer Impact]
