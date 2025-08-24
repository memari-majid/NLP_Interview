#!/usr/bin/env python3
"""
Comprehensive NLP Interview Flashcard Generator
Creates extensive flashcard coverage across all major NLP topics.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Set

class ComprehensiveNLPGenerator:
    """Generate comprehensive NLP flashcards covering all major topics."""
    
    def __init__(self):
        self.comprehensive_topics = self._define_comprehensive_topics()
        
    def _define_comprehensive_topics(self) -> Dict:
        """Define comprehensive NLP topics with extensive flashcard coverage."""
        return {
            'fundamentals': {
                'name': 'NLP Fundamentals',
                'description': 'Core concepts and foundations of Natural Language Processing',
                'cards': [
                    {
                        'front': 'What is Natural Language Processing (NLP)?',
                        'back': '''<b>Definition:</b> Field of AI that enables computers to understand, interpret, and generate human language<br>
<b>Goal:</b> Bridge gap between human communication and computer understanding<br>
<b>Applications:</b> Translation, chatbots, sentiment analysis, search<br>
<b>Challenges:</b> Ambiguity, context, cultural nuances, grammar variations'''
                    },
                    {
                        'front': 'What are the main challenges in NLP?',
                        'back': '''<b>Ambiguity:</b> Words/sentences with multiple meanings<br>
<b>Context dependency:</b> Meaning changes based on context<br>
<b>Grammar variations:</b> Informal language, typos, dialects<br>
<b>Cultural nuances:</b> Idioms, sarcasm, cultural references'''
                    },
                    {
                        'front': 'What is the curse of dimensionality in NLP?',
                        'back': '''<b>Problem:</b> Sparse data in high-dimensional spaces<br>
<b>Effect:</b> Most vectors are nearly orthogonal<br>
<b>Solution:</b> Embeddings reduce to dense, lower dimensions<br>
<b>Example:</b> One-hot vectors vs 300d word embeddings'''
                    },
                    {
                        'front': 'What are the levels of NLP analysis?',
                        'back': '''<b>Lexical:</b> Words, morphemes, character analysis<br>
<b>Syntactic:</b> Grammar, parsing, sentence structure<br>
<b>Semantic:</b> Meaning, word sense, relationships<br>
<b>Pragmatic:</b> Context, intent, discourse analysis'''
                    }
                ]
            },
            'preprocessing': {
                'name': 'Text Preprocessing',
                'description': 'Text cleaning, normalization, and preparation techniques',
                'cards': [
                    {
                        'front': 'Why do we need tokenization?',
                        'back': '''<b>Purpose:</b> Split text into meaningful units for processing<br>
<b>Challenges:</b> Punctuation, contractions, out-of-vocabulary words<br>
<b>Modern:</b> Subword tokenization (BPE, SentencePiece)<br>
<b>Benefit:</b> Balance between vocabulary size and coverage'''
                    },
                    {
                        'front': 'What is the difference between stemming and lemmatization?',
                        'back': '''<b>Stemming:</b> Remove suffixes using rules (faster, cruder)<br>
<b>Lemmatization:</b> Reduce to dictionary form using morphology<br>
<b>Example:</b> "running" → "run" (stem) vs "running" → "run" (lemma)<br>
<b>Trade-off:</b> Speed vs accuracy'''
                    },
                    {
                        'front': 'What are stop words and when should you remove them?',
                        'back': '''<b>Definition:</b> Common words like "the", "is", "and"<br>
<b>Remove when:</b> Focus on content words, reduce noise<br>
<b>Keep when:</b> Syntax matters, phrase detection, sentiment<br>
<b>Context:</b> "not good" vs "good" - "not" is crucial'''
                    },
                    {
                        'front': 'What is Part-of-Speech (POS) tagging?',
                        'back': '''<b>Definition:</b> Assign grammatical categories to words<br>
<b>Tags:</b> Noun, verb, adjective, adverb, etc.<br>
<b>Uses:</b> Parsing, NER, information extraction<br>
<b>Challenge:</b> Ambiguous words ("book" as noun vs verb)'''
                    },
                    {
                        'front': 'What is Named Entity Recognition (NER)?',
                        'back': '''<b>Definition:</b> Identify and classify named entities in text<br>
<b>Types:</b> Person, location, organization, date, money<br>
<b>Approaches:</b> Rule-based, ML-based, deep learning<br>
<b>Applications:</b> Information extraction, question answering'''
                    }
                ]
            },
            'language_modeling': {
                'name': 'Language Modeling',
                'description': 'Statistical and neural approaches to language modeling',
                'cards': [
                    {
                        'front': 'What are n-grams and why are they useful?',
                        'back': '''<b>Definition:</b> Contiguous sequences of n tokens<br>
<b>Examples:</b> Unigrams (words), bigrams (word pairs), trigrams<br>
<b>Use:</b> Language modeling, feature extraction<br>
<b>Trade-off:</b> Higher n captures more context but increases sparsity'''
                    },
                    {
                        'front': 'What is the problem with n-gram language models?',
                        'back': '''<b>Sparsity:</b> Many n-grams never seen in training<br>
<b>Smoothing needed:</b> Assign probability to unseen sequences<br>
<b>Limited context:</b> Fixed window size<br>
<b>Curse of dimensionality:</b> Exponential growth with vocabulary'''
                    },
                    {
                        'front': 'What is perplexity in language modeling?',
                        'back': '''<b>Definition:</b> Measure of how well a model predicts text<br>
<b>Formula:</b> PP = exp(-1/N × Σlog P(wᵢ))<br>
<b>Interpretation:</b> Average branching factor<br>
<b>Lower is better:</b> Model is less "perplexed" by the text'''
                    },
                    {
                        'front': 'How do neural language models work?',
                        'back': '''<b>Approach:</b> Use neural networks to predict next word<br>
<b>Advantages:</b> Continuous representations, longer context<br>
<b>Architectures:</b> RNN, LSTM, Transformer<br>
<b>Training:</b> Maximize likelihood of next word prediction'''
                    }
                ]
            },
            'word_representations': {
                'name': 'Word Representations',
                'description': 'From one-hot vectors to contextual embeddings',
                'cards': [
                    {
                        'front': 'What is TF-IDF?',
                        'back': '''<b>Intuition:</b> Weight terms by frequency and rarity across documents<br>
<b>Formula:</b> TF-IDF = TF × log(N/df)<br>
<b>Symbols:</b> TF=term freq, N=total docs, df=docs with term<br>
<b>Use:</b> Information retrieval and document similarity'''
                    },
                    {
                        'front': 'What are word embeddings?',
                        'back': '''<b>Intuition:</b> Map discrete tokens to dense continuous vectors<br>
<b>Property:</b> Similar words have similar vectors<br>
<b>Training:</b> Word2Vec (skip-gram/CBOW) or context prediction<br>
<b>Benefit:</b> Captures semantic relationships in vector space'''
                    },
                    {
                        'front': 'How does Word2Vec work?',
                        'back': '''<b>Skip-gram:</b> Predict context words from target word<br>
<b>CBOW:</b> Predict target word from context words<br>
<b>Training:</b> Negative sampling or hierarchical softmax<br>
<b>Result:</b> Dense vectors where similar words cluster together'''
                    },
                    {
                        'front': 'What is the difference between Word2Vec and GloVe?',
                        'back': '''<b>Word2Vec:</b> Local context window, predictive model<br>
<b>GloVe:</b> Global co-occurrence statistics, count-based<br>
<b>Training:</b> Word2Vec uses SGD, GloVe uses matrix factorization<br>
<b>Performance:</b> Often similar, GloVe more interpretable'''
                    },
                    {
                        'front': 'Word2Vec vs contextual embeddings?',
                        'back': '''<b>Word2Vec:</b> One vector per word type (fast, fixed)<br>
<b>Contextual:</b> Different vectors per word occurrence<br>
<b>Example:</b> "bank" (river) vs "bank" (money)<br>
<b>Trade-off:</b> Contextual more accurate but computationally expensive'''
                    },
                    {
                        'front': 'What is FastText and how does it differ from Word2Vec?',
                        'back': '''<b>Innovation:</b> Learns representations for character n-grams<br>
<b>Advantage:</b> Handles out-of-vocabulary words<br>
<b>Subword info:</b> "unhappy" = "un" + "happy" + word<br>
<b>Use case:</b> Morphologically rich languages, rare words'''
                    }
                ]
            },
            'similarity_metrics': {
                'name': 'Similarity & Distance Metrics',
                'description': 'Measuring semantic and syntactic similarity between texts',
                'cards': [
                    {
                        'front': 'What is cosine similarity?',
                        'back': '''<b>Intuition:</b> Measure angle between vectors (direction, not magnitude)<br>
<b>Formula:</b> cos(θ) = A·B / (||A|| × ||B||)<br>
<b>Range:</b> [-1,1] where 1=same direction, 0=orthogonal<br>
<b>Use:</b> Document similarity, recommendation systems'''
                    },
                    {
                        'front': 'Cosine vs Euclidean similarity?',
                        'back': '''<b>Cosine:</b> Measures direction (angle) - length independent<br>
<b>Euclidean:</b> Measures distance - sensitive to magnitude<br>
<b>Text:</b> Cosine better (document length varies)<br>
<b>Images:</b> Euclidean often used (pixel intensities matter)'''
                    },
                    {
                        'front': 'What is Jaccard similarity?',
                        'back': '''<b>Formula:</b> |A ∩ B| / |A ∪ B|<br>
<b>Interpretation:</b> Overlap divided by union<br>
<b>Range:</b> [0,1] where 1=identical sets<br>
<b>Use:</b> Set similarity, document deduplication'''
                    },
                    {
                        'front': 'What is edit distance (Levenshtein)?',
                        'back': '''<b>Definition:</b> Minimum edits to transform one string to another<br>
<b>Operations:</b> Insert, delete, substitute<br>
<b>Algorithm:</b> Dynamic programming O(mn)<br>
<b>Applications:</b> Spell correction, fuzzy matching'''
                    }
                ]
            },
            'syntax_parsing': {
                'name': 'Syntactic Analysis',
                'description': 'Grammar, parsing, and sentence structure analysis',
                'cards': [
                    {
                        'front': 'What is dependency parsing?',
                        'back': '''<b>Definition:</b> Analyze grammatical dependencies between words<br>
<b>Structure:</b> Directed graph with head-dependent relations<br>
<b>Relations:</b> subject, object, modifier, etc.<br>
<b>Applications:</b> Information extraction, question answering'''
                    },
                    {
                        'front': 'What is constituency parsing?',
                        'back': '''<b>Definition:</b> Break sentences into nested constituents<br>
<b>Structure:</b> Tree with phrases as internal nodes<br>
<b>Phrases:</b> Noun phrase (NP), verb phrase (VP), etc.<br>
<b>Grammar:</b> Context-free grammar rules'''
                    },
                    {
                        'front': 'Dependency vs constituency parsing?',
                        'back': '''<b>Dependency:</b> Word-to-word relations, more flexible<br>
<b>Constituency:</b> Phrase structure, hierarchical<br>
<b>Use cases:</b> Dependency for IE, constituency for syntax<br>
<b>Complexity:</b> Dependency often simpler to implement'''
                    }
                ]
            },
            'semantic_analysis': {
                'name': 'Semantic Analysis',
                'description': 'Understanding meaning, context, and relationships',
                'cards': [
                    {
                        'front': 'What is word sense disambiguation (WSD)?',
                        'back': '''<b>Problem:</b> Words have multiple meanings in different contexts<br>
<b>Example:</b> "bank" (financial) vs "bank" (river)<br>
<b>Approaches:</b> Supervised ML, knowledge-based, embeddings<br>
<b>Evaluation:</b> Accuracy on sense-annotated corpora'''
                    },
                    {
                        'front': 'What is semantic role labeling (SRL)?',
                        'back': '''<b>Definition:</b> Identify who did what to whom<br>
<b>Roles:</b> Agent, patient, instrument, location, time<br>
<b>Example:</b> "John (agent) ate (predicate) pizza (patient)"<br>
<b>Applications:</b> Question answering, information extraction'''
                    },
                    {
                        'front': 'What is coreference resolution?',
                        'back': '''<b>Definition:</b> Link pronouns/mentions to correct entities<br>
<b>Example:</b> "John went to store. He bought milk." (He → John)<br>
<b>Challenges:</b> Ambiguous pronouns, long-distance references<br>
<b>Methods:</b> Rule-based, ML, neural approaches'''
                    }
                ]
            },
            'nlp_tasks': {
                'name': 'Core NLP Tasks',
                'description': 'Text classification, sentiment analysis, and applications',
                'cards': [
                    {
                        'front': 'What is text classification?',
                        'back': '''<b>Definition:</b> Assign predefined categories to text documents<br>
<b>Types:</b> Binary, multi-class, multi-label<br>
<b>Approaches:</b> Naive Bayes, SVM, neural networks<br>
<b>Applications:</b> Spam detection, topic classification, intent'''
                    },
                    {
                        'front': 'What is sentiment analysis?',
                        'back': '''<b>Definition:</b> Determine emotional tone or opinion in text<br>
<b>Levels:</b> Document, sentence, aspect-based<br>
<b>Approaches:</b> Lexicon-based, ML, deep learning<br>
<b>Challenges:</b> Sarcasm, context, domain adaptation'''
                    },
                    {
                        'front': 'What are the main approaches to machine translation?',
                        'back': '''<b>Rule-based:</b> Linguistic rules and dictionaries<br>
<b>Statistical:</b> Phrase-based, alignment models<br>
<b>Neural:</b> Seq2seq, attention, transformer<br>
<b>Current:</b> Transformer-based models dominate'''
                    },
                    {
                        'front': 'What is text summarization?',
                        'back': '''<b>Extractive:</b> Select important sentences from original<br>
<b>Abstractive:</b> Generate new sentences expressing key ideas<br>
<b>Evaluation:</b> ROUGE scores, human evaluation<br>
<b>Challenges:</b> Coherence, factual accuracy, length control'''
                    },
                    {
                        'front': 'What is question answering (QA)?',
                        'back': '''<b>Types:</b> Extractive, generative, knowledge-based<br>
<b>Extractive:</b> Find answer span in given passage<br>
<b>Generative:</b> Generate answer from understanding<br>
<b>Datasets:</b> SQuAD, Natural Questions, MS MARCO'''
                    }
                ]
            },
            'modern_architectures': {
                'name': 'Modern Architectures',
                'description': 'Attention, transformers, and pre-trained models',
                'cards': [
                    {
                        'front': 'What is self-attention?',
                        'back': '''<b>Intuition:</b> Weigh tokens by relevance to each other in sequence<br>
<b>Formula:</b> Attention(Q,K,V) = softmax(QK^T/√d_k)V<br>
<b>Symbols:</b> Q=query, K=key, V=value, d_k=key dimension<br>
<b>Why:</b> Enables parallel computation and long-range dependencies'''
                    },
                    {
                        'front': 'Why use attention over RNNs?',
                        'back': '''<b>Parallelization:</b> All positions computed simultaneously<br>
<b>Long-range:</b> Direct connections between any two positions<br>
<b>Interpretability:</b> Attention weights show model focus<br>
<b>Performance:</b> Better at capturing dependencies in long sequences'''
                    },
                    {
                        'front': 'What are the key components of transformer architecture?',
                        'back': '''<b>Multi-head attention:</b> Parallel attention mechanisms<br>
<b>Position encoding:</b> Inject sequence order information<br>
<b>Layer norm + residual:</b> Training stability<br>
<b>Feed-forward:</b> Non-linear transformations per position'''
                    },
                    {
                        'front': 'What is BERT and how does it work?',
                        'back': '''<b>Architecture:</b> Bidirectional encoder-only transformer<br>
<b>Training:</b> Masked language modeling + next sentence prediction<br>
<b>Innovation:</b> Bidirectional context understanding<br>
<b>Usage:</b> Pre-train then fine-tune for downstream tasks'''
                    },
                    {
                        'front': 'What is GPT and how does it differ from BERT?',
                        'back': '''<b>GPT:</b> Autoregressive, decoder-only, left-to-right<br>
<b>BERT:</b> Bidirectional, encoder-only, masked training<br>
<b>Use:</b> GPT for generation, BERT for understanding<br>
<b>Training:</b> GPT predicts next token, BERT fills masks'''
                    },
                    {
                        'front': 'What is transfer learning in NLP?',
                        'back': '''<b>Approach:</b> Pre-train on large corpus, fine-tune on task<br>
<b>Benefits:</b> Better performance, less task-specific data needed<br>
<b>Examples:</b> BERT, GPT, RoBERTa, T5<br>
<b>Process:</b> Unsupervised pre-training → supervised fine-tuning'''
                    }
                ]
            },
            'evaluation_metrics': {
                'name': 'Evaluation & Metrics',
                'description': 'How to measure NLP model performance',
                'cards': [
                    {
                        'front': 'What is the difference between precision and recall?',
                        'back': '''<b>Precision:</b> TP/(TP+FP) - accuracy of positive predictions<br>
<b>Recall:</b> TP/(TP+FN) - fraction of positives found<br>
<b>Trade-off:</b> High precision → low recall, vice versa<br>
<b>F1:</b> Harmonic mean balances both metrics'''
                    },
                    {
                        'front': 'What is BLEU score?',
                        'back': '''<b>Use:</b> Evaluate machine translation quality<br>
<b>Method:</b> N-gram overlap between prediction and reference<br>
<b>Range:</b> 0-100, higher is better<br>
<b>Limitations:</b> Doesn't consider semantics, multiple valid translations'''
                    },
                    {
                        'front': 'What is ROUGE score?',
                        'back': '''<b>Use:</b> Evaluate text summarization quality<br>
<b>Types:</b> ROUGE-N (n-gram), ROUGE-L (LCS), ROUGE-S (skip-gram)<br>
<b>Method:</b> Overlap between generated and reference summaries<br>
<b>Interpretation:</b> Higher scores indicate better summaries'''
                    },
                    {
                        'front': 'How do you evaluate language models?',
                        'back': '''<b>Perplexity:</b> How well model predicts held-out text<br>
<b>Downstream tasks:</b> Performance on specific applications<br>
<b>Human evaluation:</b> Fluency, coherence, relevance<br>
<b>Intrinsic vs extrinsic:</b> Model-specific vs task-specific'''
                    }
                ]
            },
            'advanced_topics': {
                'name': 'Advanced Topics',
                'description': 'Cutting-edge techniques and considerations',
                'cards': [
                    {
                        'front': 'What is few-shot learning in NLP?',
                        'back': '''<b>Definition:</b> Learn new tasks with very few examples<br>
<b>Approach:</b> Use pre-trained models + prompt engineering<br>
<b>Examples:</b> GPT-3 with task descriptions in prompts<br>
<b>Benefits:</b> Rapid adaptation, minimal task-specific data'''
                    },
                    {
                        'front': 'What is prompt engineering?',
                        'back': '''<b>Definition:</b> Design input prompts to elicit desired behavior<br>
<b>Techniques:</b> Few-shot examples, instruction tuning<br>
<b>Importance:</b> Can dramatically affect model performance<br>
<b>Challenges:</b> Brittle, model-dependent, hard to optimize'''
                    },
                    {
                        'front': 'What are common biases in NLP models?',
                        'back': '''<b>Gender bias:</b> Associating professions with genders<br>
<b>Racial bias:</b> Different sentiment for different groups<br>
<b>Cultural bias:</b> Western-centric training data<br>
<b>Mitigation:</b> Diverse data, bias testing, debiasing techniques'''
                    },
                    {
                        'front': 'What is model interpretability in NLP?',
                        'back': '''<b>Attention visualization:</b> Which words model focuses on<br>
<b>Gradient-based:</b> Input attribution methods<br>
<b>Probing tasks:</b> What linguistic knowledge models learn<br>
<b>Local explanations:</b> LIME, SHAP for individual predictions'''
                    }
                ]
            }
        }
    
    def create_card(self, front: str, back: str, topic: str, card_type: str, deck_uuid: str) -> Dict:
        """Create a single flashcard with proper metadata."""
        content_hash = hashlib.md5(f"{front}{back}".encode()).hexdigest()[:13]
        
        return {
            "__type__": "Note",
            "fields": [front, back, topic, card_type],
            "guid": f"nlp_{content_hash}",
            "note_model_uuid": f"nlp-{deck_uuid}-model",
            "tags": [topic.lower().replace(' ', '_').replace('&', ''), card_type]
        }
    
    def create_deck_structure(self, deck_key: str, deck_info: Dict, cards: List[Dict]) -> Dict:
        """Create deck structure for a specific category."""
        deck_uuid = f"nlp-{deck_key}-deck"
        model_uuid = f"nlp-{deck_key}-model"
        config_uuid = f"nlp-{deck_key}-config"
        
        return {
            "__type__": "Deck",
            "children": [],
            "crowdanki_uuid": deck_uuid,
            "deck_config_uuid": config_uuid,
            "deck_configurations": [{
                "__type__": "DeckConfig",
                "autoplay": True,
                "crowdanki_uuid": config_uuid,
                "dyn": False,
                "name": deck_info['name'],
                "new": {
                    "delays": [1, 10],
                    "initialFactor": 2500,
                    "ints": [1, 4, 7],
                    "order": 1,
                    "perDay": 20
                },
                "rev": {
                    "ease4": 1.3,
                    "hardFactor": 1.2,
                    "ivlFct": 1.0,
                    "maxIvl": 36500,
                    "perDay": 100
                }
            }],
            "desc": deck_info['description'],
            "dyn": 0,
            "extendNew": 10,
            "extendRev": 50,
            "media_files": [],
            "name": deck_info['name'],
            "note_models": [{
                "__type__": "NoteModel",
                "crowdanki_uuid": model_uuid,
                "css": """
.card {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 18px;
    line-height: 1.5;
    color: #2c3e50;
    background: #f8f9fa;
    text-align: left;
    padding: 20px;
    border-radius: 8px;
    max-width: 100%;
    margin: 0 auto;
}

@media (max-width: 480px) {
    .card {
        font-size: 16px;
        padding: 15px;
        line-height: 1.4;
    }
}

b, strong {
    color: #2980b9;
    font-weight: 600;
}

.metadata {
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px solid #bdc3c7;
    font-size: 12px;
    color: #7f8c8d;
    text-align: center;
}
""",
                "flds": [
                    {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20},
                    {"name": "Back", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20},
                    {"name": "Topic", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 16},
                    {"name": "Type", "ord": 3, "sticky": False, "rtl": False, "font": "Arial", "size": 14}
                ],
                "latexPost": "\\end{document}",
                "latexPre": "\\documentclass[12pt]{article}\\special{papersize=3in,5in}\\usepackage{amssymb,amsmath}\\pagestyle{empty}\\setlength{\\parindent}{0in}\\begin{document}",
                "name": f"NLP {deck_info['name']}",
                "req": [[0, "all", [0]]],
                "sortf": 0,
                "tags": [],
                "tmpls": [{
                    "afmt": "{{FrontSide}}<hr id=answer>{{Back}}<br><br><div class='metadata'><span style='background: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600;'>{{Topic}}</span> • <span style='background: #95a5a6; color: white; padding: 4px 8px; border-radius: 4px; margin-left: 8px;'>{{Type}}</span></div>",
                    "bafmt": "",
                    "bqfmt": "",
                    "did": None,
                    "name": "Card 1",
                    "ord": 0,
                    "qfmt": "{{Front}}"
                }],
                "type": 0,
                "vers": []
            }],
            "notes": cards
        }
    
    def generate_all_comprehensive_decks(self):
        """Generate comprehensive NLP flashcard decks."""
        # Clean up existing directories first
        import shutil
        for deck_key in self.comprehensive_topics.keys():
            deck_dir = f"NLP_{deck_key.title()}"
            if os.path.exists(deck_dir):
                shutil.rmtree(deck_dir)
        
        total_cards = 0
        deck_summary = []
        
        for deck_key, deck_info in self.comprehensive_topics.items():
            # Create directory
            deck_dir = Path(f"NLP_{deck_key.title()}")
            deck_dir.mkdir(exist_ok=True)
            
            # Generate cards
            cards = []
            for card_data in deck_info['cards']:
                card = self.create_card(
                    front=card_data['front'],
                    back=card_data['back'],
                    topic=deck_info['name'],
                    card_type='concept',
                    deck_uuid=deck_key
                )
                cards.append(card)
            
            # Create deck structure
            deck_data = self.create_deck_structure(deck_key, deck_info, cards)
            
            # Save deck file with proper naming
            deck_filename = f"NLP_{deck_key.title()}.json"
            output_file = deck_dir / deck_filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(deck_data, f, indent=2, ensure_ascii=False)
            
            # Create individual README
            self._create_deck_readme(deck_dir, deck_info, len(cards))
            
            deck_summary.append({
                'name': deck_info['name'],
                'folder': str(deck_dir),
                'cards': len(cards),
                'description': deck_info['description']
            })
            total_cards += len(cards)
        
        # Create master README
        self._create_master_readme(deck_summary, total_cards)
        
        return deck_summary, total_cards
    
    def _create_deck_readme(self, deck_dir: Path, deck_info: Dict, card_count: int):
        """Create README for individual deck."""
        readme_content = f"""# 🎯 {deck_info['name']} Flashcards

## 📚 Overview
{deck_info['description']}. **{card_count} essential cards** covering comprehensive knowledge in this area.

## 🚀 Import to Anki
1. **Copy this entire folder** to your computer
2. **Open Anki** → `File` → `CrowdAnki: Import from disk`
3. **Select this folder** (`{deck_dir.name}`)
4. **Import** - deck will appear as "{deck_info['name']}"

## 📱 Study Settings
- **New cards**: 15-20 per day
- **Review time**: ~20 seconds per card
- **Total study**: 10-15 minutes daily

## 🎯 Learning Focus
Master the essential concepts in {deck_info['name'].lower()} that are commonly tested in NLP interviews.
"""
        
        readme_file = deck_dir / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _create_master_readme(self, deck_summary: List[Dict], total_cards: int):
        """Create comprehensive master README."""
        readme_content = f"""# 🎯 Comprehensive NLP Interview Flashcard Collection

## 📚 Overview
**{total_cards} flashcards** across **{len(deck_summary)} specialized decks** covering all major NLP topics for comprehensive interview preparation.

## 📊 Complete Deck Collection

| Deck | Cards | Focus Area |
|------|-------|------------|
"""
        
        for deck in deck_summary:
            readme_content += f"| **[{deck['folder']}/]({deck['folder']}/)** | {deck['cards']} | {deck['description']} |\n"
        
        readme_content += f"""

**Total Coverage**: {total_cards} cards spanning the entire NLP field

## 🚀 Import Strategy

### Option 1: Complete Coverage (Recommended for Comprehensive Prep)
Import all {len(deck_summary)} decks for complete NLP knowledge coverage:
1. Copy all `NLP_*` folders to your computer
2. Import each folder separately using CrowdAnki
3. Study systematically through all topics

### Option 2: Targeted Preparation
Choose specific areas based on your interview focus:
- **Fundamentals** + **Preprocessing** for entry-level roles
- **Modern Architectures** + **Advanced Topics** for senior positions
- **Evaluation** + **NLP Tasks** for applied ML roles

## 📚 Recommended Study Path

### Phase 1: Foundation (Weeks 1-2)
Start with core concepts and text processing fundamentals.

### Phase 2: Traditional Methods (Weeks 2-3)  
Learn language modeling and word representations.

### Phase 3: Advanced Understanding (Weeks 3-4)
Master syntactic and semantic analysis techniques.

### Phase 4: Modern Techniques (Weeks 4-5)
Study attention mechanisms and transformer architectures.

### Phase 5: Evaluation & Applications (Weeks 5-6)
Learn evaluation metrics and practical applications.

## 📱 Daily Study Schedule
- **New cards**: 20-30 across all active decks
- **Reviews**: 80-120 per day (Anki manages this)
- **Total time**: 30-45 minutes daily
- **Active decks**: 2-3 decks simultaneously for depth

## ✅ Quality Features
- **Research-backed content** covering all major NLP areas
- **Interview-focused** questions and explanations
- **15-20 second review time** per card
- **Mobile-optimized** for study anywhere
- **Comprehensive coverage** of the entire NLP field

---

**🎯 Ready for comprehensive NLP interview preparation? Start with NLP_Fundamentals and work through systematically!**
"""
        
        with open('Comprehensive_NLP_Flashcards_README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)

def main():
    """Generate comprehensive NLP flashcard collection."""
    generator = ComprehensiveNLPGenerator()
    deck_summary, total_cards = generator.generate_all_comprehensive_decks()
    
    print(f"\n✅ Comprehensive NLP Interview Flashcards Created!")
    print(f"📊 Total: {total_cards} cards across {len(deck_summary)} specialized decks")
    print(f"\n🎯 Decks Created:")
    
    for deck in deck_summary:
        print(f"  📚 {deck['name']}: {deck['cards']} cards")
        print(f"     Folder: {deck['folder']}/")
        print(f"     Focus: {deck['description']}")
        print()
    
    print(f"🚀 Complete NLP coverage for thorough interview preparation!")
    print(f"📖 See Comprehensive_NLP_Flashcards_README.md for study strategy")

if __name__ == "__main__":
    main()
