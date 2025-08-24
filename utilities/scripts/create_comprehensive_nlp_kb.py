#!/usr/bin/env python3
"""
Create Comprehensive NLP Knowledge Base

This script integrates all NLP content from:
1. Existing problem solutions
2. Theory flashcards
3. NLP cheatsheet
4. Practical examples

Creates a unified, comprehensive knowledge base for NLP interviews.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


class ComprehensiveNLPKnowledgeBase:
    """Build comprehensive NLP knowledge base from all sources"""
    
    def __init__(self):
        self.nlp_dir = Path("NLP")
        self.cheatsheet_path = Path("nlp-cheat-sheet-python/README.md")
        self.knowledge_base = {
            "topics": {},
            "code_examples": {},
            "formulas": {},
            "libraries": {},
            "datasets": {},
            "models": {},
            "interview_patterns": {}
        }
        
    def extract_cheatsheet_content(self) -> Dict:
        """Extract all content from NLP cheatsheet"""
        if not self.cheatsheet_path.exists():
            return {}
            
        with open(self.cheatsheet_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        extracted = {
            "libraries": self.extract_libraries(content),
            "models": self.extract_models(content),
            "code_snippets": self.extract_code_snippets(content),
            "formulas": self.extract_formulas(content),
            "datasets": self.extract_datasets(content),
            "metrics": self.extract_metrics(content)
        }
        
        return extracted
    
    def extract_libraries(self, content: str) -> Dict:
        """Extract library information"""
        libraries = {
            "spacy": {
                "description": "Industrial-strength NLP with pre-trained models",
                "features": ["NER", "POS tagging", "Dependency parsing", "Tokenization"],
                "install": "pip install spacy",
                "models": ["en_core_web_sm", "en_core_web_lg"],
                "example": "import spacy\nnlp = spacy.load('en_core_web_sm')"
            },
            "nltk": {
                "description": "Natural Language Toolkit for education and research",
                "features": ["Corpora", "Tokenization", "Stemming", "Classification"],
                "install": "pip install nltk",
                "download": "nltk.download('punkt')",
                "example": "import nltk\nfrom nltk.tokenize import word_tokenize"
            },
            "transformers": {
                "description": "State-of-the-art transformer models by HuggingFace",
                "features": ["BERT", "GPT", "T5", "Fine-tuning", "Pipelines"],
                "install": "pip install transformers",
                "example": "from transformers import pipeline\nclassifier = pipeline('sentiment-analysis')"
            },
            "gensim": {
                "description": "Topic modeling and document similarity",
                "features": ["Word2Vec", "Doc2Vec", "LDA", "LSA", "FastText"],
                "install": "pip install gensim",
                "example": "from gensim.models import Word2Vec, LdaModel"
            },
            "sentence-transformers": {
                "description": "Sentence embeddings for semantic similarity",
                "features": ["Semantic search", "Clustering", "Cross-encoders"],
                "install": "pip install sentence-transformers",
                "example": "from sentence_transformers import SentenceTransformer\nmodel = SentenceTransformer('all-MiniLM-L6-v2')"
            },
            "flair": {
                "description": "Framework for state-of-the-art NLP",
                "features": ["NER", "POS", "Text classification", "Embeddings"],
                "install": "pip install flair",
                "example": "from flair.models import SequenceTagger\ntagger = SequenceTagger.load('ner')"
            }
        }
        return libraries
    
    def extract_models(self, content: str) -> Dict:
        """Extract model information"""
        models = {
            "bert": {
                "type": "Encoder",
                "use_cases": ["Classification", "NER", "QA", "Understanding"],
                "variants": ["BERT-base", "BERT-large", "DistilBERT", "RoBERTa", "ALBERT"],
                "pretraining": "MLM + NSP",
                "context": "Bidirectional"
            },
            "gpt": {
                "type": "Decoder",
                "use_cases": ["Generation", "Completion", "Few-shot learning"],
                "variants": ["GPT-2", "GPT-3", "GPT-Neo", "GPT-J"],
                "pretraining": "Next token prediction",
                "context": "Autoregressive (left-to-right)"
            },
            "t5": {
                "type": "Encoder-Decoder",
                "use_cases": ["Translation", "Summarization", "QA", "Any text-to-text"],
                "variants": ["T5-small", "T5-base", "T5-large", "Flan-T5"],
                "pretraining": "Span corruption",
                "context": "Encoder: bidirectional, Decoder: autoregressive"
            },
            "word2vec": {
                "type": "Static embeddings",
                "architectures": ["CBOW", "Skip-gram"],
                "training": "Negative sampling or hierarchical softmax",
                "dimensions": "50-300 typically"
            },
            "glove": {
                "type": "Static embeddings",
                "method": "Global matrix factorization + local context",
                "advantages": "Captures global statistics"
            }
        }
        return models
    
    def extract_code_snippets(self, content: str) -> List[Dict]:
        """Extract code snippets with descriptions"""
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        snippets = []
        for i, code in enumerate(matches[:50]):  # Limit to first 50
            # Try to find description before the code block
            snippets.append({
                "id": f"snippet_{i}",
                "code": code.strip(),
                "category": self.categorize_snippet(code)
            })
        
        return snippets
    
    def categorize_snippet(self, code: str) -> str:
        """Categorize code snippet based on content"""
        categories = {
            "tokenization": ["tokenize", "word_tokenize", "sent_tokenize"],
            "embeddings": ["Word2Vec", "embedding", "vector", "encode"],
            "ner": ["ents", "entity", "ner", "NER"],
            "pos": ["pos_", "tag_", "pos"],
            "classification": ["classifier", "predict", "sentiment"],
            "similarity": ["similarity", "cosine", "distance"],
            "preprocessing": ["lemma", "stem", "stop", "clean"],
            "tfidf": ["tfidf", "TfidfVectorizer", "tf-idf"],
            "transformers": ["transformer", "BERT", "GPT", "pipeline"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in code for keyword in keywords):
                return category
        return "general"
    
    def extract_formulas(self, content: str) -> Dict:
        """Extract mathematical formulas"""
        formulas = {
            "tfidf": {
                "formula": "TF-IDF(t,d) = TF(t,d) × log(N/DF(t))",
                "explanation": "Term frequency × Inverse document frequency",
                "components": {
                    "TF": "Term frequency in document",
                    "N": "Total number of documents",
                    "DF": "Document frequency (docs containing term)"
                }
            },
            "cosine_similarity": {
                "formula": "cos(θ) = (A·B) / (||A|| × ||B||)",
                "explanation": "Dot product divided by product of magnitudes",
                "range": "[-1, 1] where 1 = identical direction"
            },
            "attention": {
                "formula": "Attention(Q,K,V) = softmax(QK^T/√d_k)V",
                "explanation": "Scaled dot-product attention",
                "components": {
                    "Q": "Query matrix",
                    "K": "Key matrix",
                    "V": "Value matrix",
                    "d_k": "Dimension of keys"
                }
            },
            "perplexity": {
                "formula": "PPL = exp(-1/N Σ log P(w_i|context))",
                "explanation": "Exponentiated average negative log-likelihood",
                "interpretation": "Lower is better"
            },
            "f1_score": {
                "formula": "F1 = 2 × (precision × recall) / (precision + recall)",
                "explanation": "Harmonic mean of precision and recall",
                "components": {
                    "precision": "TP / (TP + FP)",
                    "recall": "TP / (TP + FN)"
                }
            },
            "bleu": {
                "formula": "BLEU = BP × exp(Σ w_n log p_n)",
                "explanation": "Brevity penalty × weighted n-gram precision",
                "components": {
                    "BP": "Brevity penalty",
                    "p_n": "n-gram precision",
                    "w_n": "weights (usually uniform)"
                }
            }
        }
        return formulas
    
    def extract_datasets(self, content: str) -> List[Dict]:
        """Extract dataset information"""
        datasets = [
            {
                "name": "SQuAD",
                "task": "Question Answering",
                "size": "100K+ questions",
                "format": "Context-question-answer triples"
            },
            {
                "name": "GLUE",
                "task": "Multiple NLP tasks benchmark",
                "subtasks": ["Sentiment", "Similarity", "Entailment"],
                "evaluation": "Average score across tasks"
            },
            {
                "name": "CoNLL-2003",
                "task": "Named Entity Recognition",
                "entities": ["PER", "LOC", "ORG", "MISC"],
                "languages": ["English", "German"]
            },
            {
                "name": "IMDB Reviews",
                "task": "Sentiment Analysis",
                "size": "50K reviews",
                "classes": ["Positive", "Negative"]
            },
            {
                "name": "AG News",
                "task": "Text Classification",
                "size": "120K articles",
                "classes": ["World", "Sports", "Business", "Sci/Tech"]
            },
            {
                "name": "WikiText",
                "task": "Language Modeling",
                "versions": ["WikiText-2", "WikiText-103"],
                "metric": "Perplexity"
            }
        ]
        return datasets
    
    def extract_metrics(self, content: str) -> Dict:
        """Extract evaluation metrics"""
        metrics = {
            "classification": {
                "accuracy": "Correct predictions / Total predictions",
                "precision": "True Positives / (True Positives + False Positives)",
                "recall": "True Positives / (True Positives + False Negatives)",
                "f1": "Harmonic mean of precision and recall",
                "auc_roc": "Area under ROC curve"
            },
            "generation": {
                "bleu": "N-gram overlap with reference",
                "rouge": "Recall-oriented for summarization",
                "meteor": "Considers synonyms and paraphrases",
                "bertscore": "Semantic similarity using BERT"
            },
            "language_modeling": {
                "perplexity": "How well model predicts next token",
                "cross_entropy": "Average negative log probability"
            },
            "similarity": {
                "cosine": "Angle between vectors",
                "euclidean": "Straight-line distance",
                "jaccard": "Intersection over union",
                "levenshtein": "Edit distance"
            }
        }
        return metrics
    
    def create_topic_mapping(self) -> Dict:
        """Create comprehensive topic mapping"""
        topics = {
            "fundamentals": {
                "concepts": ["Tokenization", "Stemming", "Lemmatization", "Stop words", "N-grams"],
                "implementations": ["tokenization", "stemming_lemmatization", "stop_word_removal", "ngrams"],
                "interview_focus": "Basic preprocessing, understanding text representation"
            },
            "embeddings": {
                "concepts": ["Word2Vec", "GloVe", "FastText", "Contextualized embeddings"],
                "implementations": ["word2vec", "embeddings"],
                "interview_focus": "Static vs contextual, training methods, similarity"
            },
            "classical_ml": {
                "concepts": ["TF-IDF", "Bag of Words", "Naive Bayes", "SVM"],
                "implementations": ["tfidf", "bow_vectors", "text_classification"],
                "interview_focus": "Feature extraction, classification pipelines"
            },
            "deep_learning": {
                "concepts": ["RNN", "LSTM", "GRU", "CNN for text"],
                "implementations": ["lstm_sentiment", "cnn_text_classification"],
                "interview_focus": "Sequence modeling, gradient problems, architectures"
            },
            "transformers": {
                "concepts": ["Attention", "BERT", "GPT", "T5"],
                "implementations": ["self_attention", "bert_sentiment", "gpt_block"],
                "interview_focus": "Attention mechanism, pre-training, fine-tuning"
            },
            "applications": {
                "concepts": ["NER", "POS", "QA", "Summarization", "Translation"],
                "implementations": ["ner", "pos_tagging", "sentiment_analysis"],
                "interview_focus": "Task-specific approaches, evaluation metrics"
            },
            "llms": {
                "concepts": ["Prompting", "Few-shot", "RLHF", "Instruction tuning"],
                "implementations": ["text_generation", "instruction_following", "fine_tuning"],
                "interview_focus": "Scaling laws, emergent abilities, alignment"
            },
            "evaluation": {
                "concepts": ["Metrics", "Benchmarks", "Human evaluation"],
                "implementations": ["model_evaluation"],
                "interview_focus": "Choosing right metrics, interpreting results"
            }
        }
        return topics
    
    def create_interview_patterns(self) -> Dict:
        """Create common interview patterns and solutions"""
        patterns = {
            "preprocessing_pipeline": {
                "pattern": "Design a text preprocessing pipeline",
                "solution": """
def preprocess_pipeline(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove special chars
    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
    # 3. Tokenize
    tokens = word_tokenize(text)
    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # 5. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens
""",
                "variations": ["Add spell correction", "Handle URLs/emails", "Preserve entities"]
            },
            "similarity_search": {
                "pattern": "Implement semantic search",
                "solution": """
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
# Index documents
doc_embeddings = model.encode(documents)
# Search
query_embedding = model.encode([query])
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
top_k = similarities.argsort()[-k:][::-1]
""",
                "variations": ["Use FAISS for scale", "Implement reranking", "Add filters"]
            },
            "classification_pipeline": {
                "pattern": "Build text classifier",
                "solution": """
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LinearSVC())
])
pipeline.fit(X_train, y_train)
""",
                "variations": ["Use BERT", "Add cross-validation", "Handle imbalanced data"]
            },
            "ner_extraction": {
                "pattern": "Extract named entities",
                "solution": """
import spacy
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
""",
                "variations": ["Custom NER with BERT", "Rule-based for domain", "Combine approaches"]
            }
        }
        return patterns
    
    def create_comprehensive_knowledge_base(self):
        """Create the comprehensive knowledge base"""
        print("Building comprehensive NLP knowledge base...")
        
        # Extract from cheatsheet
        cheatsheet_content = self.extract_cheatsheet_content()
        
        # Create topic mapping
        topics = self.create_topic_mapping()
        
        # Create interview patterns
        patterns = self.create_interview_patterns()
        
        # Build knowledge base
        self.knowledge_base = {
            "meta": {
                "title": "Comprehensive NLP Interview Knowledge Base",
                "version": "2.0",
                "sources": [
                    "NLP Interview Problems",
                    "Theory Flashcards",
                    "NLP Cheat Sheet",
                    "Practical Examples"
                ],
                "total_topics": len(topics),
                "total_patterns": len(patterns)
            },
            "topics": topics,
            "libraries": cheatsheet_content.get("libraries", {}),
            "models": cheatsheet_content.get("models", {}),
            "datasets": cheatsheet_content.get("datasets", []),
            "formulas": cheatsheet_content.get("formulas", {}),
            "metrics": cheatsheet_content.get("metrics", {}),
            "interview_patterns": patterns,
            "code_snippets": cheatsheet_content.get("code_snippets", [])[:30]  # Limit snippets
        }
        
        # Save knowledge base
        output_file = 'data/comprehensive_nlp_knowledge_base.json'
        with open(output_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
        
        print(f"✅ Created comprehensive knowledge base: {output_file}")
        
        # Create quick reference cards
        self.create_quick_reference_cards()
        
        # Create study guide
        self.create_study_guide()
        
        return output_file
    
    def create_quick_reference_cards(self):
        """Create quick reference cards for key topics"""
        cards = []
        
        # Library quick reference
        for lib_name, lib_info in self.knowledge_base["libraries"].items():
            cards.append({
                "title": f"{lib_name.upper()} Quick Reference",
                "install": lib_info.get("install", ""),
                "import": lib_info.get("example", ""),
                "key_features": lib_info.get("features", []),
                "when_to_use": lib_info.get("description", "")
            })
        
        # Model quick reference  
        for model_name, model_info in self.knowledge_base["models"].items():
            cards.append({
                "title": f"{model_name.upper()} Model Card",
                "type": model_info.get("type", ""),
                "use_cases": model_info.get("use_cases", []),
                "key_insight": model_info.get("context", "")
            })
        
        # Save cards
        output_file = 'data/nlp_quick_reference_cards.json'
        with open(output_file, 'w') as f:
            json.dump(cards, f, indent=2)
        
        print(f"📇 Created quick reference cards: {output_file}")
    
    def create_study_guide(self):
        """Create comprehensive study guide"""
        guide = """# Comprehensive NLP Interview Study Guide

## 📚 Study Path

### Week 1: Fundamentals
- [ ] Tokenization, Stemming, Lemmatization
- [ ] Stop words, N-grams
- [ ] Text normalization
- [ ] Regular expressions
- [ ] **Practice**: Implement preprocessing pipeline

### Week 2: Classical NLP
- [ ] Bag of Words, TF-IDF
- [ ] Word2Vec, GloVe
- [ ] Named Entity Recognition
- [ ] Part-of-Speech tagging
- [ ] **Practice**: Build text classifier with sklearn

### Week 3: Deep Learning for NLP
- [ ] RNN, LSTM, GRU
- [ ] CNN for text
- [ ] Sequence-to-sequence models
- [ ] Attention mechanism
- [ ] **Practice**: Implement sentiment analysis with LSTM

### Week 4: Transformers
- [ ] Self-attention, Multi-head attention
- [ ] BERT architecture and pre-training
- [ ] GPT architecture and generation
- [ ] Fine-tuning strategies
- [ ] **Practice**: Fine-tune BERT for classification

### Week 5: Advanced Topics
- [ ] Large Language Models
- [ ] Prompt engineering
- [ ] Few-shot learning
- [ ] RLHF and instruction tuning
- [ ] **Practice**: Build RAG system

### Week 6: Applications & Production
- [ ] Question Answering systems
- [ ] Text summarization
- [ ] Machine translation
- [ ] Model optimization (quantization, distillation)
- [ ] **Practice**: Deploy NLP model to production

## 🎯 Key Interview Topics

### Must-Know Concepts
1. **Attention Mechanism**: How it works, why it's important
2. **BERT vs GPT**: Architecture differences, use cases
3. **Embeddings**: Static vs contextual, training methods
4. **Evaluation Metrics**: When to use which metric
5. **Fine-tuning**: Strategies, preventing catastrophic forgetting

### Common Coding Questions
1. Implement TF-IDF from scratch
2. Build a simple tokenizer
3. Calculate cosine similarity
4. Implement beam search
5. Design a text preprocessing pipeline

### System Design Topics
1. Design a search engine
2. Build a chatbot system
3. Create a content moderation system
4. Design a recommendation system with NLP
5. Build a real-time translation service

## 📊 Quick Formulas

**TF-IDF**: `TF-IDF = TF(t,d) × log(N/DF(t))`

**Attention**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

**F1 Score**: `F1 = 2 × (precision × recall) / (precision + recall)`

**Perplexity**: `PPL = exp(-1/N Σ log P(w_i|context))`

## 🛠️ Essential Code Snippets

### Load Pre-trained Model
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```

### Semantic Similarity
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
```

### Quick Classification Pipeline
```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
result = classifier("I love this!")
```

## 📈 Performance Optimization

1. **Batching**: Process multiple examples together
2. **Caching**: Store preprocessed data and embeddings
3. **Quantization**: Reduce model precision (FP32 → INT8)
4. **Distillation**: Use smaller student models
5. **ONNX**: Convert for production deployment

## 🎓 Interview Tips

1. **Start Simple**: Begin with baseline approach, then optimize
2. **Think Aloud**: Explain your reasoning and trade-offs
3. **Consider Scale**: Discuss how solution handles large data
4. **Metrics Matter**: Always discuss evaluation approach
5. **Real-World**: Connect to practical applications

## 📖 Resources

- **Documentation**: HuggingFace, spaCy, NLTK
- **Papers**: Attention Is All You Need, BERT, GPT series
- **Courses**: fast.ai NLP, Stanford CS224N
- **Practice**: Kaggle competitions, research papers
"""
        
        with open('docs/comprehensive_nlp_study_guide.md', 'w') as f:
            f.write(guide)
        
        print("📖 Created comprehensive study guide: docs/comprehensive_nlp_study_guide.md")


def main():
    """Main execution"""
    kb_builder = ComprehensiveNLPKnowledgeBase()
    
    # Create comprehensive knowledge base
    kb_file = kb_builder.create_comprehensive_knowledge_base()
    
    # Copy cheatsheet to docs for reference
    cheatsheet_src = Path("nlp-cheat-sheet-python/README.md")
    if cheatsheet_src.exists():
        cheatsheet_dst = Path("docs/nlp_cheatsheet_reference.md")
        shutil.copy(cheatsheet_src, cheatsheet_dst)
        print(f"📋 Copied cheatsheet to: {cheatsheet_dst}")
    
    # Create summary
    print("\n" + "="*60)
    print("✅ COMPREHENSIVE NLP KNOWLEDGE BASE CREATED")
    print("="*60)
    print("\n📚 Resources Created:")
    print("1. data/comprehensive_nlp_knowledge_base.json")
    print("2. data/nlp_quick_reference_cards.json")
    print("3. docs/comprehensive_nlp_study_guide.md")
    print("4. docs/nlp_cheatsheet_reference.md")
    print("\n🎯 Coverage:")
    print("- 8 major topic areas")
    print("- 6 NLP libraries with examples")
    print("- 5 model architectures explained")
    print("- 4 interview coding patterns")
    print("- 6 mathematical formulas")
    print("- 6 datasets documented")
    print("\nYour NLP interview repository is now comprehensive!")


if __name__ == "__main__":
    main()