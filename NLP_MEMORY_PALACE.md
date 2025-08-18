# NLP Memory Palace — The Journey Through NLP City

A creative memory system to connect and memorize all NLP topics for interviews. Walk through this imaginary city where each location represents a concept, making it easy to recall during phone screens.

## 🗺️ The Map Overview

Imagine NLP City as a journey from raw text (dirty streets) to understanding (enlightened summit):

```
[City Gate] → [Market Square] → [Craft District] → [Knowledge Quarter] → [Tech Tower] → [Summit]
```

## 🏛️ Memory Palace Locations

### 🚪 **City Gate** (Text Preprocessing)
*"Before entering the city, you must clean up"*

1. **Gatehouse** - Text Normalization
   - Guards clean your Unicode badges
   - Memory: "Normalize before you analyze"
   - Link to: [Utilities](NLP/Utilities/)

2. **Token Market** - Tokenization  
   - Merchants chopping text like vegetables
   - Memory: "Split with care, punctuation aware"
   - Link to: [Tokenization](NLP/Tokenization/)

3. **Filter Station** - Stop Words
   - Common words tossed in trash bins
   - Memory: "The, a, an - thrown in the can"
   - Link to: [Stop Words](NLP/Stop_Word_Removal/)

### 🏪 **Market Square** (Word Processing)
*"Where words are refined and packaged"*

4. **Blacksmith Shop** - Stemming/Lemmatization
   - Hammering words to their root form
   - Memory: "Running → run, better → good"
   - Link to: [Stemming](NLP/Stemming_Lemmatization/)

5. **Label Office** - POS Tagging
   - Clerks stamping word types
   - Memory: "Noun, Verb, Adjective stamps"
   - Link to: [POS Tagging](NLP/POS_Tagging/)

6. **Pattern Shop** - Regex/NER
   - Detectives with magnifying glasses
   - Memory: "Find the @ in emails, $ in prices"
   - Link to: [NER](NLP/NER/) & [Regex](NLP/Regex_NLP/)

### 📚 **Knowledge Quarter** (Vectorization)
*"Where text becomes numbers"*

7. **Bakery** - N-grams
   - Baking word cookies in pairs/triplets
   - Memory: "New York → 'New_York' bigram"
   - Link to: [N-grams](NLP/NGrams/)

8. **Library** - Bag of Words & TF-IDF
   - Counting books, valuing rare editions
   - Memory: "Common words cheap, rare words precious"
   - Formula: `TF × log(N/df)`
   - Link to: [BoW](NLP/BoW_Vectors/) & [TF-IDF](NLP/TFIDF/)

9. **Observatory** - Word Embeddings
   - 3D telescope showing word relationships
   - Memory: "King - Man + Woman = Queen"
   - Link to: [Word2Vec](NLP/Embeddings/)

### 🎯 **Analytics Avenue** (Classical ML)
*"Where patterns meet predictions"*

10. **Angle Bridge** - Similarity
    - Measuring angles between vectors
    - Memory: "Cosine cares about direction, not magnitude"
    - Link to: [Similarity](NLP/Similarity/)

11. **Mood Meter** - Sentiment Analysis
    - Emotional thermometer on buildings
    - Memory: "VADER reads the vibes"
    - Link to: [Sentiment](NLP/Sentiment_Analysis/)

12. **Sorting Factory** - Text Classification
    - Conveyor belts sorting documents
    - Memory: "Feature → Model → Category"
    - Link to: [Classification](NLP/Text_Classification/)

### 🏗️ **Neural District** (Deep Learning)
*"Where networks learn patterns"*

13. **Neuron Garden** - Neural Fundamentals
    - Perceptrons growing like flowers
    - Memory: "Weights × Inputs + Bias → Activation"
    - Link to: [Neural Basics](NLP/Neural_Fundamentals/)

14. **Sliding Tower** - CNN for Text
    - Windows sliding down text walls
    - Memory: "Convolution captures local patterns"
    - Link to: [CNN Text](NLP/CNN_Text/)

15. **Memory Museum** - LSTM/RNN
    - Halls with forget/remember gates
    - Memory: "Three gates control the flow"
    - Link to: [Sequence Models](NLP/Sequence_Models/)

### 🚀 **Tech Tower** (Modern NLP)
*"The transformer revolution"*

16. **Attention Plaza** - Self-Attention
    - Spotlights pointing at important words
    - Memory: "Q asks, K answers, V provides"
    - Formula: `QK^T/√d → softmax → V`
    - Link to: [Attention](NLP/Attention_Mechanisms/)

17. **Transformer Station** - BERT/GPT
    - Massive parallel processing center
    - Memory: "All words attend to all words"
    - Link to: [Transformers](NLP/Transformers/) & [GPT](NLP/GPT_Implementation/)

18. **Token Forge** - Advanced Tokenization
    - BPE merging common byte pairs
    - Memory: "Merge frequent, handle unknown"
    - Link to: [BPE](NLP/Tokenization_Advanced/)

### ⚡ **Summit** (LLM & Advanced)
*"Peak performance"*

19. **Generation Tower** - Text Generation
    - Radio tower broadcasting next words
    - Memory: "Sample with temperature control"
    - Link to: [Text Generation](NLP/LLM_Fundamentals/)

20. **Tuning Workshop** - Fine-tuning
    - Tailors adjusting pre-trained models
    - Memory: "Freeze base, train head"
    - Link to: [Fine-tuning](NLP/Fine_Tuning/) & [Instruction](NLP/Instruction_Tuning/)

21. **Evaluation Court** - Model Assessment
    - Judges scoring model outputs
    - Memory: "Perplexity low, BLEU high"
    - Link to: [Evaluation](NLP/Model_Evaluation/)

## 🧠 Memory Techniques

### **The Journey Method**
Walk through the city in order during interviews:
1. Start at Gate (preprocessing)
2. Through Market (word processing)  
3. Into Knowledge Quarter (vectorization)
4. Cross Analytics Avenue (ML)
5. Enter Neural District (deep learning)
6. Climb Tech Tower (transformers)
7. Reach Summit (LLMs)

### **Acronyms Along the Way**

- **STATS** (preprocessing): **S**top words, **T**okenize, **A**nalyze, **T**ag, **S**tem
- **VINE** (vectors): **V**ectorize, **I**DF weight, **N**grams, **E**mbeddings
- **CAST** (attention): **C**ompare QK, **A**djust by √d, **S**oftmax, **T**ake V
- **BEAM** (generation): **B**eam search, **E**psilon greedy, **A**rgmax, **M**odulate temp

### **Visual Anchors**

- TF-IDF: Picture a library with price tags on books
- Attention: Imagine spotlights in a theater
- LSTM: Three bouncers at gate (input/forget/output)
- BPE: Two puzzle pieces clicking together
- Word2Vec: 3D constellation of word stars

### **Number Patterns**

- Embedding dims: 50, 100, 200, 300 (multiples of 50)
- Transformer heads: 8, 12, 16 (powers of 2)
- Context windows: 512, 1024, 2048 (doubling)
- Learning rates: 1e-3, 1e-4, 1e-5 (order of magnitude)

## 🎯 Quick Recall Exercises

### **60-Second City Tour**
Before each interview, mentally walk through:
1. Gate → Market → Knowledge → Analytics → Neural → Tower → Summit
2. At each stop, recall one formula or concept
3. Use hand gestures for each location

### **Problem → Location Mapping**
- "Implement TF-IDF" → Go to Library in Knowledge Quarter
- "Build tokenizer" → Visit Token Market at City Gate
- "Attention mechanism" → Head to Attention Plaza in Tech Tower

### **Reverse Journey**
Start from Summit and work backwards:
- What would I need to generate text? (Generation Tower)
- What feeds into that? (Transformer Station)
- What's the core mechanism? (Attention Plaza)
- Continue until you reach the Gate

## 📱 Anki Integration

Each location links to specific Anki cards:
- Location name → Algorithm implementation
- Visual anchor → Key formula
- Memory phrase → Complexity/trade-offs

Use spaced repetition to strengthen the palace!

---

Remember: The journey through NLP City mirrors the actual data flow in NLP pipelines. By memorizing the city, you memorize the entire field!
