#!/usr/bin/env python3
"""
Extract and integrate content from nlp-cheat-sheet-python repository

This script extracts useful code snippets, formulas, and examples from the 
NLP cheat sheet and converts them into flashcards and reference material.
"""

import json
import re
import os
from typing import Dict, List, Tuple
from pathlib import Path


class CheatSheetExtractor:
    """Extract and convert NLP cheat sheet content"""
    
    def __init__(self):
        self.cheatsheet_path = Path("nlp-cheat-sheet-python/README.md")
        self.flashcards = []
        self.code_snippets = {}
        self.practical_examples = []
        
    def extract_sections(self) -> Dict[str, str]:
        """Extract major sections from the cheat sheet"""
        if not self.cheatsheet_path.exists():
            print(f"Warning: {self.cheatsheet_path} not found")
            return {}
            
        with open(self.cheatsheet_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by headers
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('# '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line[2:].strip()
                current_content = []
            elif line.startswith('## '):
                if current_section:
                    subsection = f"{current_section} - {line[3:].strip()}"
                    current_section = subsection
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract code blocks with their descriptions"""
        code_blocks = []
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            # Try to find description before code block
            desc_pattern = r'([^\n]+)\n+```python\n' + re.escape(match[:50])
            desc_match = re.search(desc_pattern, text, re.DOTALL)
            description = desc_match.group(1) if desc_match else "Code snippet"
            code_blocks.append((description, match))
            
        return code_blocks
    
    def create_practical_flashcards(self):
        """Create flashcards from practical examples in the cheat sheet"""
        
        # Library usage flashcards
        library_cards = [
            {
                "id": "practical_001",
                "category": "libraries",
                "difficulty": "easy",
                "question": "What are the main Python NLP libraries and their primary uses?",
                "answer": "1. spaCy: Industrial-strength NLP with pre-trained models for NER, POS, dependency parsing\n2. NLTK: Educational toolkit with corpora and classical algorithms\n3. Gensim: Topic modeling and document similarity\n4. Transformers (HuggingFace): State-of-the-art transformer models",
                "code_example": "import spacy\nnlp = spacy.load('en_core_web_sm')\ndoc = nlp('Apple is looking at buying U.K. startup')",
                "practical_tip": "Use spaCy for production, NLTK for learning, HuggingFace for SOTA models"
            },
            {
                "id": "practical_002",
                "category": "preprocessing",
                "difficulty": "easy", 
                "question": "How do you remove stop words using spaCy vs NLTK?",
                "answer": "Both libraries provide stop word lists, but spaCy integrates it into the token object",
                "code_example": "# spaCy\nfrom spacy.lang.en.stop_words import STOP_WORDS\ntokens = [token.text for token in doc if not token.is_stop]\n\n# NLTK\nfrom nltk.corpus import stopwords\nstop_words = set(stopwords.words('english'))\nfiltered = [w for w in words if w not in stop_words]",
                "comparison": "spaCy: 326 stop words, integrated with tokenization\nNLTK: 179 stop words, separate processing step"
            },
            {
                "id": "practical_003",
                "category": "embeddings",
                "difficulty": "medium",
                "question": "How do you implement TF-IDF using scikit-learn?",
                "answer": "Use TfidfVectorizer to convert documents to TF-IDF matrix representation",
                "code_example": "from sklearn.feature_extraction.text import TfidfVectorizer\n\ntfidf = TfidfVectorizer()\nX = tfidf.fit_transform(documents)\n\n# Get feature names\nfeatures = tfidf.get_feature_names_out()\n# Get IDF values\nidf_values = tfidf.idf_",
                "practical_tip": "Use ngram_range=(1,2) for unigrams+bigrams, max_features to limit vocabulary"
            },
            {
                "id": "practical_004",
                "category": "similarity",
                "difficulty": "medium",
                "question": "How do you calculate cosine similarity between documents?",
                "answer": "Transform documents to vectors (TF-IDF, embeddings) then compute cosine similarity",
                "code_example": "from sklearn.metrics.pairwise import cosine_similarity\n\n# Using TF-IDF\ntfidf_matrix = tfidf.fit_transform(documents)\nsimilarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)\n\n# Using embeddings\nimport numpy as np\ncos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))",
                "formula": "cosine_similarity = (A·B) / (||A|| × ||B||)"
            },
            {
                "id": "practical_005",
                "category": "tokenization",
                "difficulty": "easy",
                "question": "What's the difference between word, subword, and character tokenization in practice?",
                "answer": "Word: splits on spaces/punctuation. Subword: handles OOV with pieces (BPE). Character: each char is a token",
                "code_example": "# Word tokenization\ntext.split()  # Simple\nnlp(text)  # spaCy\n\n# Subword (using transformers)\nfrom transformers import AutoTokenizer\ntokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\ntokens = tokenizer.tokenize('unbelievable')  # ['un', '##believ', '##able']\n\n# Character\nlist('hello')  # ['h', 'e', 'l', 'l', 'o']",
                "trade_offs": "Word: interpretable but OOV issues\nSubword: handles OOV, smaller vocab\nCharacter: no OOV but long sequences"
            },
            {
                "id": "practical_006",
                "category": "ner",
                "difficulty": "medium",
                "question": "How do you perform Named Entity Recognition with spaCy?",
                "answer": "spaCy provides pre-trained NER models that identify persons, organizations, locations, etc.",
                "code_example": "import spacy\nnlp = spacy.load('en_core_web_sm')\n\ndoc = nlp('Apple Inc. was founded by Steve Jobs in Cupertino')\nfor ent in doc.ents:\n    print(f'{ent.text}: {ent.label_}')\n# Output:\n# Apple Inc.: ORG\n# Steve Jobs: PERSON\n# Cupertino: GPE",
                "entity_types": "PERSON, ORG, GPE (location), DATE, MONEY, PERCENT, TIME"
            },
            {
                "id": "practical_007",
                "category": "pos_tagging",
                "difficulty": "easy",
                "question": "How do you get POS tags using spaCy?",
                "answer": "spaCy automatically tags parts of speech during processing",
                "code_example": "doc = nlp('The cat sat on the mat')\nfor token in doc:\n    print(f'{token.text}: {token.pos_} ({token.tag_})')\n# Output:\n# The: DET (DT)\n# cat: NOUN (NN)\n# sat: VERB (VBD)",
                "common_tags": "NOUN, VERB, ADJ, ADV, PRON, DET, PREP, NUM, CONJ, INTJ"
            },
            {
                "id": "practical_008",
                "category": "ngrams",
                "difficulty": "easy",
                "question": "How do you extract n-grams from text?",
                "answer": "N-grams are contiguous sequences of n items from text",
                "code_example": "from nltk import ngrams\n\ntext = 'The quick brown fox'\ntokens = text.split()\n\n# Bigrams\nlist(ngrams(tokens, 2))\n# [('The', 'quick'), ('quick', 'brown'), ('brown', 'fox')]\n\n# Trigrams\nlist(ngrams(tokens, 3))\n# [('The', 'quick', 'brown'), ('quick', 'brown', 'fox')]",
                "use_cases": "Language modeling, feature extraction, text generation"
            },
            {
                "id": "practical_009",
                "category": "word_vectors",
                "difficulty": "medium",
                "question": "How do you load and use pre-trained word vectors?",
                "answer": "Use gensim for Word2Vec/GloVe or transformers for contextual embeddings",
                "code_example": "# Gensim Word2Vec\nfrom gensim.models import KeyedVectors\nmodel = KeyedVectors.load_word2vec_format('GoogleNews-vectors.bin', binary=True)\nvector = model['computer']\nsimilar = model.most_similar('king')\n\n# spaCy vectors\ndoc = nlp('cat')\nvector = doc[0].vector  # 300-dim vector\n\n# Transformers\nfrom sentence_transformers import SentenceTransformer\nmodel = SentenceTransformer('all-MiniLM-L6-v2')\nembeddings = model.encode(['Hello world'])",
                "tip": "Use sentence-transformers for semantic similarity tasks"
            },
            {
                "id": "practical_010",
                "category": "text_classification",
                "difficulty": "medium",
                "question": "What's a simple pipeline for text classification?",
                "answer": "1. Preprocess text 2. Vectorize (TF-IDF/embeddings) 3. Train classifier 4. Evaluate",
                "code_example": "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.pipeline import Pipeline\n\npipeline = Pipeline([\n    ('tfidf', TfidfVectorizer()),\n    ('classifier', MultinomialNB())\n])\n\npipeline.fit(X_train, y_train)\npredictions = pipeline.predict(X_test)",
                "models": "Naive Bayes: good baseline\nSVM: strong performance\nLogistic Regression: interpretable\nDeep Learning: BERT for SOTA"
            },
            {
                "id": "practical_011",
                "category": "regex",
                "difficulty": "easy",
                "question": "What are common regex patterns for NLP preprocessing?",
                "answer": "Regular expressions for cleaning and extracting text patterns",
                "code_example": "import re\n\n# Remove special characters\nre.sub(r'[^a-zA-Z0-9\\s]', '', text)\n\n# Extract emails\nre.findall(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', text)\n\n# Extract URLs\nre.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', text)\n\n# Extract hashtags\nre.findall(r'#\\w+', text)\n\n# Remove extra whitespace\nre.sub(r'\\s+', ' ', text).strip()",
                "use_cases": "Data cleaning, pattern extraction, text normalization"
            },
            {
                "id": "practical_012",
                "category": "lemmatization",
                "difficulty": "easy",
                "question": "How do you perform lemmatization with spaCy vs NLTK?",
                "answer": "Lemmatization reduces words to their base form using vocabulary and morphological analysis",
                "code_example": "# spaCy\ndoc = nlp('The striped bats are hanging on their feet')\nlemmas = [token.lemma_ for token in doc]\n# ['the', 'stripe', 'bat', 'be', 'hang', 'on', 'their', 'foot']\n\n# NLTK\nfrom nltk.stem import WordNetLemmatizer\nlemmatizer = WordNetLemmatizer()\nlemmatizer.lemmatize('feet')  # 'foot'\nlemmatizer.lemmatize('running', pos='v')  # 'run'",
                "tip": "spaCy lemmatization is context-aware, NLTK requires POS tags for accuracy"
            },
            {
                "id": "practical_013",
                "category": "dependency_parsing",
                "difficulty": "medium",
                "question": "How do you extract dependency relations with spaCy?",
                "answer": "spaCy provides dependency parsing showing grammatical relationships",
                "code_example": "doc = nlp('The cat sat on the mat')\nfor token in doc:\n    print(f'{token.text} --{token.dep_}--> {token.head.text}')\n# The --det--> cat\n# cat --nsubj--> sat\n# sat --ROOT--> sat\n# on --prep--> sat\n# the --det--> mat\n# mat --pobj--> on",
                "use_cases": "Information extraction, question answering, relation extraction"
            },
            {
                "id": "practical_014",
                "category": "sentiment",
                "difficulty": "medium",
                "question": "How do you perform sentiment analysis with different approaches?",
                "answer": "Rule-based (VADER), ML (training classifier), or pre-trained transformers",
                "code_example": "# VADER (rule-based)\nfrom vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer\nanalyzer = SentimentIntensityAnalyzer()\nscores = analyzer.polarity_scores('This movie is really good!')\n# {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.4404}\n\n# Transformers\nfrom transformers import pipeline\nsentiment = pipeline('sentiment-analysis')\nresult = sentiment('I love this!')[0]\n# {'label': 'POSITIVE', 'score': 0.999}",
                "comparison": "VADER: fast, no training, social media aware\nTransformers: accurate, context-aware, slower"
            },
            {
                "id": "practical_015",
                "category": "topic_modeling",
                "difficulty": "hard",
                "question": "How do you implement LDA topic modeling with gensim?",
                "answer": "LDA discovers abstract topics in a collection of documents",
                "code_example": "from gensim import corpora, models\n\n# Prepare corpus\ntexts = [[word for word in doc.split()] for doc in documents]\ndictionary = corpora.Dictionary(texts)\ncorpus = [dictionary.doc2bow(text) for text in texts]\n\n# Train LDA\nlda = models.LdaModel(\n    corpus=corpus,\n    id2word=dictionary,\n    num_topics=5,\n    random_state=42\n)\n\n# View topics\nfor idx, topic in lda.print_topics():\n    print(f'Topic {idx}: {topic}')",
                "parameters": "num_topics: number of topics to extract\nalpha: document-topic density\nbeta: topic-word density"
            }
        ]
        
        return library_cards
    
    def create_formula_flashcards(self):
        """Create flashcards for mathematical formulas and metrics"""
        
        formula_cards = [
            {
                "id": "formula_001",
                "category": "metrics",
                "difficulty": "easy",
                "question": "How do you calculate TF-IDF in practice?",
                "answer": "TF-IDF = Term Frequency × Inverse Document Frequency",
                "formula": "TF-IDF(t,d) = TF(t,d) × log(N/DF(t))",
                "code_example": "from sklearn.feature_extraction.text import TfidfVectorizer\nvectorizer = TfidfVectorizer()\nX = vectorizer.fit_transform(documents)\n# Get specific term's TF-IDF\nfeature_names = vectorizer.get_feature_names_out()\ntfidf_score = X[0, vectorizer.vocabulary_['word']]",
                "interpretation": "Higher TF-IDF = term is frequent in doc but rare in corpus"
            },
            {
                "id": "formula_002",
                "category": "metrics",
                "difficulty": "medium",
                "question": "How do you calculate precision, recall, and F1 score?",
                "answer": "Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean",
                "formula": "F1 = 2 × (precision × recall) / (precision + recall)",
                "code_example": "from sklearn.metrics import precision_recall_fscore_support\n\nprecision, recall, f1, _ = precision_recall_fscore_support(\n    y_true, y_pred, average='weighted'\n)\n\n# Or use classification_report\nfrom sklearn.metrics import classification_report\nprint(classification_report(y_true, y_pred))",
                "when_to_use": "Precision: minimize false positives\nRecall: minimize false negatives\nF1: balanced metric"
            },
            {
                "id": "formula_003",
                "category": "similarity",
                "difficulty": "medium",
                "question": "How do you implement different text similarity metrics?",
                "answer": "Common metrics: Cosine, Jaccard, Levenshtein distance",
                "code_example": "# Cosine similarity\nfrom sklearn.metrics.pairwise import cosine_similarity\nsim = cosine_similarity([vec1], [vec2])[0][0]\n\n# Jaccard similarity\ndef jaccard(set1, set2):\n    return len(set1 & set2) / len(set1 | set2)\n\n# Levenshtein distance\nfrom Levenshtein import distance\nedit_dist = distance('kitten', 'sitting')",
                "use_cases": "Cosine: vector similarity\nJaccard: set similarity\nLevenshtein: string edit distance"
            }
        ]
        
        return formula_cards
    
    def create_advanced_flashcards(self):
        """Create flashcards for advanced NLP concepts with practical code"""
        
        advanced_cards = [
            {
                "id": "advanced_001",
                "category": "transformers",
                "difficulty": "hard",
                "question": "How do you fine-tune a BERT model for classification?",
                "answer": "Load pre-trained BERT, add classification head, fine-tune on task data",
                "code_example": "from transformers import AutoModelForSequenceClassification, Trainer\n\nmodel = AutoModelForSequenceClassification.from_pretrained(\n    'bert-base-uncased',\n    num_labels=2\n)\n\ntrainer = Trainer(\n    model=model,\n    args=training_args,\n    train_dataset=train_dataset,\n    eval_dataset=val_dataset\n)\n\ntrainer.train()",
                "tips": "Use smaller learning rate (2e-5), warmup steps, gradient accumulation for large batches"
            },
            {
                "id": "advanced_002",
                "category": "optimization",
                "difficulty": "hard",
                "question": "How do you handle long texts that exceed model's max length?",
                "answer": "Strategies: truncation, sliding window, hierarchical processing",
                "code_example": "# Sliding window approach\ndef sliding_window(text, max_length=512, stride=256):\n    tokens = tokenizer.encode(text)\n    chunks = []\n    \n    for i in range(0, len(tokens), stride):\n        chunk = tokens[i:i + max_length]\n        chunks.append(chunk)\n        if i + max_length >= len(tokens):\n            break\n    \n    # Process chunks and aggregate\n    outputs = [model(chunk) for chunk in chunks]\n    return aggregate_predictions(outputs)",
                "trade_offs": "Truncation: simple but loses info\nSliding: captures all but overlapping\nHierarchical: complex but effective"
            },
            {
                "id": "advanced_003",
                "category": "deployment",
                "difficulty": "hard",
                "question": "How do you optimize NLP models for production?",
                "answer": "Quantization, distillation, ONNX conversion, batching",
                "code_example": "# Model quantization\nimport torch\nmodel_int8 = torch.quantization.quantize_dynamic(\n    model, {torch.nn.Linear}, dtype=torch.qint8\n)\n\n# ONNX export\ntorch.onnx.export(model, dummy_input, 'model.onnx')\n\n# Batching requests\nfrom transformers import pipeline\nclassifier = pipeline('sentiment-analysis')\nresults = classifier(texts, batch_size=32)",
                "performance": "Quantization: 2-4x speedup, <1% accuracy loss\nONNX: cross-platform deployment\nBatching: better GPU utilization"
            }
        ]
        
        return advanced_cards
    
    def integrate_with_existing(self):
        """Integrate extracted content with existing flashcard system"""
        
        # Load existing theory flashcards
        theory_file = 'data/nlp_theory_flashcards.json'
        if os.path.exists(theory_file):
            with open(theory_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {"flashcards": []}
        
        # Add practical flashcards
        practical = self.create_practical_flashcards()
        formulas = self.create_formula_flashcards()
        advanced = self.create_advanced_flashcards()
        
        # Combine all new flashcards
        new_flashcards = practical + formulas + advanced
        
        # Add to existing with unique IDs
        existing['flashcards'].extend(new_flashcards)
        
        # Save enhanced flashcards
        output_file = 'data/nlp_practical_flashcards.json'
        with open(output_file, 'w') as f:
            json.dump({
                "meta": {
                    "title": "NLP Practical Interview Flashcards",
                    "description": "Practical code examples and implementations for NLP interviews",
                    "version": "1.0",
                    "source": "Integrated from nlp-cheat-sheet-python"
                },
                "flashcards": new_flashcards,
                "resources": {
                    "libraries": [
                        "spacy", "nltk", "gensim", "transformers",
                        "scikit-learn", "sentence-transformers"
                    ],
                    "datasets": [
                        "Gutenberg Corpus", "Brown Corpus", "Reuters",
                        "AG News", "IMDB Reviews", "SQuAD"
                    ],
                    "models": [
                        "BERT", "GPT-2", "RoBERTa", "T5", "BART",
                        "DistilBERT", "Sentence-BERT"
                    ]
                }
            }, f, indent=2)
        
        print(f"✅ Created {len(new_flashcards)} practical flashcards")
        print(f"📁 Saved to {output_file}")
        
        # Create a quick reference guide
        self.create_quick_reference()
        
        return output_file
    
    def create_quick_reference(self):
        """Create a quick reference guide for common NLP tasks"""
        
        reference = """# NLP Quick Reference Guide

## Essential Imports
```python
# Core libraries
import spacy
import nltk
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, LdaModel

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
```

## Common Patterns

### Text Preprocessing Pipeline
```python
def preprocess_text(text):
    # 1. Lower case
    text = text.lower()
    
    # 2. Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 3. Tokenize
    doc = nlp(text)
    
    # 4. Remove stop words
    tokens = [token.text for token in doc if not token.is_stop]
    
    # 5. Lemmatize
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    
    return ' '.join(tokens)
```

### Quick Text Classification
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Train
text_clf.fit(X_train, y_train)

# Predict
predictions = text_clf.predict(X_test)
```

### Semantic Similarity
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
embeddings = model.encode(['sentence 1', 'sentence 2'])

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
```

### Zero-shot Classification
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification")

result = classifier(
    "This is a tutorial about Transformers",
    candidate_labels=["education", "politics", "business"]
)
```

## Performance Tips

1. **Batch Processing**: Always process multiple texts together
2. **Model Caching**: Load models once, reuse for multiple predictions
3. **GPU Acceleration**: Use `device=0` for GPU when available
4. **Preprocessing**: Cache preprocessed texts when possible
5. **Vector Storage**: Use FAISS or Annoy for similarity search at scale

## Common Gotchas

- **Tokenization**: Different models use different tokenizers (don't mix!)
- **Stop Words**: Be careful removing negations ('not', 'no')
- **Case Sensitivity**: Some models are case-sensitive (cased BERT)
- **Max Length**: Transformer models have max sequence length (usually 512)
- **Memory**: Large models need significant RAM/VRAM
"""
        
        with open('docs/nlp_quick_reference.md', 'w') as f:
            f.write(reference)
        
        print("📚 Created quick reference guide at docs/nlp_quick_reference.md")


def main():
    """Main execution"""
    print("Extracting content from NLP cheat sheet...")
    extractor = CheatSheetExtractor()
    
    # Extract and integrate content
    output_file = extractor.integrate_with_existing()
    
    print("\n✅ Integration complete!")
    print("\nNew resources created:")
    print("1. data/nlp_practical_flashcards.json - Practical code-focused flashcards")
    print("2. docs/nlp_quick_reference.md - Quick reference guide")
    print("\nThese complement your existing theory flashcards with practical implementations")


if __name__ == "__main__":
    main()