# NLP & LLM Coding Interview Question Bank

**24 interview-ready problems with concise Python solutions. Each problem designed for 15-30 minute coding sessions.**

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

### 🆕 Large Language Models (5 problems)
- **[Self-Attention](NLP/Attention_Mechanisms/)** (25 min) - Scaled dot-product attention
- **[BPE Tokenization](NLP/Tokenization_Advanced/)** (30 min) - Subword tokenization
- **[GPT Block](NLP/GPT_Implementation/)** (30 min) - Transformer architecture
- **[LLM Fine-tuning](NLP/Fine_Tuning/)** (25 min) - Classification heads, LoRA
- **[Text Generation](NLP/LLM_Fundamentals/)** (25 min) - Sampling strategies
- **[LLM Evaluation](NLP/Model_Evaluation/)** (20 min) - Perplexity, BLEU

## 🎯 Interview Focus Areas

### **Traditional NLP** (18 problems)
Core preprocessing, classical ML approaches, feature engineering

### **Modern LLMs** (6 problems) 🆕
Attention mechanisms, transformer architecture, fine-tuning, generation

## LLM Interview Concepts

### **Architecture Understanding**
- **Self-attention mechanism** - How does it work? Why is it powerful?
- **Transformer blocks** - Layer norm, residual connections, feed-forward
- **Positional encoding** - How do transformers handle sequence order?
- **Causal masking** - Why mask future tokens in autoregressive models?

### **Tokenization for LLMs**
- **Subword tokenization** - BPE, WordPiece, SentencePiece
- **Out-of-vocabulary handling** - How subwords solve OOV problem
- **Special tokens** - [CLS], [SEP], [PAD], [UNK] usage
- **Token efficiency** - Why character-level vs word-level trade-offs matter

### **Training & Fine-tuning**
- **Transfer learning** - How to adapt pretrained models
- **Parameter-efficient fine-tuning** - LoRA, adapters, prefix tuning
- **Learning rate scheduling** - Different rates for different layers
- **Instruction tuning** - How to train models to follow instructions

### **Text Generation**
- **Autoregressive generation** - Next-token prediction approach
- **Sampling strategies** - Greedy vs random vs top-k vs top-p
- **Beam search** - Finding high-probability sequences
- **Temperature scaling** - Controlling randomness vs quality

### **Evaluation**
- **Perplexity** - Core LLM metric for language modeling
- **BLEU score** - N-gram overlap for generation tasks
- **Human evaluation** - Why automated metrics aren't enough
- **Task-specific metrics** - Classification accuracy, F1, etc.

## Interview Tips

### **LLM-Specific Tips**
- **Start with conceptual explanation** before coding
- **Draw the architecture** if possible (transformer block diagram)
- **Discuss trade-offs** (model size vs performance, efficiency vs quality)
- **Mention current trends** (instruction tuning, RLHF, parameter efficiency)

### **Common LLM Interview Questions**
1. "Implement self-attention from scratch"
2. "How would you fine-tune BERT for spam detection?"
3. "Explain the difference between GPT and BERT architectures"
4. "How do you handle very long sequences in transformers?"
5. "What's the difference between greedy and beam search?"

### **Before Coding**
- Clarify model size constraints (parameters, memory)
- Ask about evaluation metrics for the specific task
- Discuss data requirements and preprocessing needs

### **During Implementation**
- Explain each step (attention → layer norm → FFN)
- Handle edge cases (empty sequences, OOV tokens)
- Mention optimization opportunities (caching, parallelization)

## Complexity Quick Reference

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| Tokenization | O(n) | O(n) | n = text length |
| TF-IDF | O(d×v) | O(d×v) | d = docs, v = vocab |
| Word2Vec | O(T×K) | O(V×K) | T = training steps |
| **Self-Attention** | **O(n²×d)** | **O(n²)** | **n = sequence length** |
| **GPT Forward** | **O(L×n²×d)** | **O(n×d)** | **L = num layers** |
| **Beam Search** | **O(k×n×V)** | **O(k×n)** | **k = beam width** |

## Company-Specific LLM Focus

### **OpenAI/Anthropic**
- GPT architecture deep-dive
- Instruction tuning and RLHF
- Safety and alignment techniques
- Prompt engineering strategies

### **Google/DeepMind**
- BERT vs GPT trade-offs
- Transformer optimizations
- Multi-modal extensions
- Efficiency improvements

### **Meta/Hugging Face**
- Model deployment and serving
- Open-source model ecosystems
- Multi-language support
- Community model fine-tuning

### **Microsoft/Amazon**
- Enterprise LLM applications
- Cost optimization strategies
- Integration with existing systems
- Responsible AI deployment

## Study Plan

### **Week 1: Foundations** (Traditional NLP)
- Master text preprocessing and feature extraction
- Implement classical ML approaches (TF-IDF, BoW)
- Practice regex and string manipulation

### **Week 2: Neural Networks**
- Build neural networks from scratch
- Understand CNNs and LSTMs for text
- Practice sequence modeling concepts

### **Week 3: Modern LLMs** 🆕
- **Day 1-2**: Self-attention mechanism implementation
- **Day 3-4**: GPT transformer block architecture  
- **Day 5-6**: BPE tokenization and text generation
- **Day 7**: Fine-tuning strategies and evaluation

### **Week 4: Advanced Topics**
- Instruction tuning and RLHF concepts
- Parameter-efficient fine-tuning (LoRA)
- Model evaluation and deployment considerations
- Company-specific interview preparation

## Usage

Each solution is self-contained and runs independently:

```bash
# Traditional NLP
python NLP/Tokenization/tokenization_solution.py
python NLP/TFIDF/tfidf_solution.py

# Modern LLMs
python NLP/Attention_Mechanisms/self_attention_solution.py
python NLP/GPT_Implementation/gpt_block_solution.py
python NLP/LLM_Fundamentals/text_generation_solution.py
```

**Dependencies**: Most solutions use only Python stdlib + numpy for algorithmic focus.

---

## Repository Stats

- **24 NLP & LLM Topics** covered
- **48+ Files** with problems and solutions
- **15-30 minutes** per problem  
- **~10 hours** total practice time
- **Interview-optimized** format for live coding

**Perfect for 2024+ NLP interviews** covering both traditional methods and modern LLM techniques! 🚀