#!/usr/bin/env python3
"""
Optimal NLP Interview Flashcard Generator
Creates high-quality, research-backed flashcards following our design rules.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Set

class OptimalAnkiGenerator:
    """Generate optimal flashcards following research-backed principles."""
    
    def __init__(self):
        self.card_templates = self._load_card_templates()
        
    def _load_card_templates(self) -> Dict:
        """High-quality templates for each NLP concept."""
        return {
            'attention_mechanisms': {
                'concept': {
                    'front': 'What is self-attention?',
                    'back': '''<b>Intuition:</b> Weigh tokens by relevance to each other in sequence<br>
<b>Formula:</b> Attention(Q,K,V) = softmax(QK^T/√d_k)V<br>
<b>Symbols:</b> Q=query, K=key, V=value, d_k=key dimension<br>
<b>Why:</b> Enables parallel computation and long-range dependencies'''
                },
                'formula': {
                    'front': 'Write the attention formula with scaling',
                    'back': '''<b>Formula:</b> softmax(QK^T/√d_k)V<br>
<b>Symbols:</b> Q=query matrix, K=key matrix, V=value matrix<br>
<b>Scaling:</b> √d_k prevents vanishing gradients in softmax<br>
<b>Output:</b> Weighted sum of values based on attention scores'''
                },
                'comparison': {
                    'front': 'Why use attention over RNNs?',
                    'back': '''<b>Parallelization:</b> All positions computed simultaneously<br>
<b>Long-range:</b> Direct connections between any two positions<br>
<b>Interpretability:</b> Attention weights show model focus<br>
<b>Performance:</b> Better at capturing dependencies in long sequences'''
                }
            },
            'tfidf': {
                'concept': {
                    'front': 'What is TF-IDF?',
                    'back': '''<b>Intuition:</b> Weight terms by frequency and rarity across documents<br>
<b>Formula:</b> TF-IDF = TF × log(N/df)<br>
<b>Symbols:</b> TF=term freq, N=total docs, df=docs with term<br>
<b>Use:</b> Information retrieval and document similarity'''
                },
                'formula': {
                    'front': 'Write the TF-IDF formula and explain components',
                    'back': '''<b>Formula:</b> TF(t,d) × IDF(t,D) = (count/|d|) × log(N/df)<br>
<b>TF:</b> Term frequency in document (normalized by doc length)<br>
<b>IDF:</b> Inverse document frequency (rarity measure)<br>
<b>Result:</b> High for frequent terms in few documents'''
                },
                'edge_case': {
                    'front': 'Common TF-IDF edge case and fix?',
                    'back': '''<b>Problem:</b> Division by zero when df=0<br>
<b>Fix:</b> Add smoothing: log(N/(df+1))<br>
<b>Alternative:</b> Filter out terms not in vocabulary<br>
<b>Impact:</b> Prevents runtime errors in production'''
                }
            },
            'embeddings': {
                'concept': {
                    'front': 'What are word embeddings?',
                    'back': '''<b>Intuition:</b> Map discrete tokens to dense continuous vectors<br>
<b>Property:</b> Similar words have similar vectors<br>
<b>Training:</b> Word2Vec (skip-gram/CBOW) or context prediction<br>
<b>Benefit:</b> Captures semantic relationships in vector space'''
                },
                'comparison': {
                    'front': 'Word2Vec vs contextual embeddings?',
                    'back': '''<b>Word2Vec:</b> One vector per word type (fast, fixed)<br>
<b>Contextual:</b> Different vectors per word occurrence<br>
<b>Example:</b> "bank" (river) vs "bank" (money)<br>
<b>Trade-off:</b> Contextual more accurate but computationally expensive'''
                }
            },
            'transformers': {
                'concept': {
                    'front': 'Why do transformers work so well?',
                    'back': '''<b>Parallelization:</b> No sequential bottleneck like RNNs<br>
<b>Attention:</b> Direct connections between all positions<br>
<b>Scalability:</b> Performance improves with more data/compute<br>
<b>Transfer:</b> Pre-trained models adapt well to new tasks'''
                },
                'architecture': {
                    'front': 'Key components of transformer architecture?',
                    'back': '''<b>Multi-head attention:</b> Parallel attention mechanisms<br>
<b>Position encoding:</b> Inject sequence order information<br>
<b>Layer norm + residual:</b> Training stability<br>
<b>Feed-forward:</b> Non-linear transformations per position'''
                }
            },
            'similarity': {
                'concept': {
                    'front': 'What is cosine similarity?',
                    'back': '''<b>Intuition:</b> Measure angle between vectors (direction, not magnitude)<br>
<b>Formula:</b> cos(θ) = A·B / (||A|| × ||B||)<br>
<b>Range:</b> [-1,1] where 1=same direction, 0=orthogonal<br>
<b>Use:</b> Document similarity, recommendation systems'''
                },
                'comparison': {
                    'front': 'Cosine vs Euclidean similarity?',
                    'back': '''<b>Cosine:</b> Measures direction (angle) - length independent<br>
<b>Euclidean:</b> Measures distance - sensitive to magnitude<br>
<b>Text:</b> Cosine better (document length varies)<br>
<b>Images:</b> Euclidean often used (pixel intensities matter)'''
                }
            },
            'sequence_models': {
                'concept': {
                    'front': 'What are the gates in LSTM?',
                    'back': '''<b>Forget:</b> What to discard from cell state<br>
<b>Input:</b> What new information to store<br>
<b>Output:</b> What parts of cell state to output<br>
<b>Purpose:</b> Control information flow and prevent vanishing gradients'''
                }
            }
        }
    
    def create_card(self, front: str, back: str, topic: str, card_type: str) -> Dict:
        """Create a single flashcard with proper metadata."""
        content_hash = hashlib.md5(f"{front}{back}".encode()).hexdigest()[:13]
        
        return {
            "__type__": "Note",
            "fields": [front, back, topic, card_type],
            "guid": f"nlp_{content_hash}",
            "note_model_uuid": "nlp-optimal-model",
            "tags": [topic.lower().replace(' ', '_'), card_type]
        }
    
    def generate_topic_cards(self, topic_key: str, topic_name: str) -> List[Dict]:
        """Generate all cards for a specific topic."""
        cards = []
        
        if topic_key not in self.card_templates:
            return cards
            
        templates = self.card_templates[topic_key]
        
        for card_type, template in templates.items():
            card = self.create_card(
                front=template['front'],
                back=template['back'],
                topic=topic_name,
                card_type=card_type
            )
            cards.append(card)
            
        return cards
    
    def create_optimal_deck(self) -> Dict:
        """Create complete optimized deck."""
        all_cards = []
        
        # Generate cards for each topic
        topics = [
            ('attention_mechanisms', 'Attention Mechanisms'),
            ('tfidf', 'TF-IDF'),
            ('embeddings', 'Embeddings'),
            ('transformers', 'Transformers'),
            ('similarity', 'Similarity'),
            ('sequence_models', 'Sequence Models')
        ]
        
        for topic_key, topic_name in topics:
            topic_cards = self.generate_topic_cards(topic_key, topic_name)
            all_cards.extend(topic_cards)
        
        # Add some general NLP concepts
        general_cards = [
            {
                'front': 'What is the curse of dimensionality in NLP?',
                'back': '''<b>Problem:</b> Sparse data in high-dimensional spaces<br>
<b>Effect:</b> Most vectors are nearly orthogonal<br>
<b>Solution:</b> Embeddings reduce to dense, lower dimensions<br>
<b>Example:</b> One-hot vectors vs 300d word embeddings''',
                'topic': 'NLP Fundamentals',
                'type': 'concept'
            },
            {
                'front': 'Why do we need tokenization?',
                'back': '''<b>Purpose:</b> Split text into meaningful units for processing<br>
<b>Challenges:</b> Punctuation, contractions, out-of-vocabulary words<br>
<b>Modern:</b> Subword tokenization (BPE, SentencePiece)<br>
<b>Benefit:</b> Balance between vocabulary size and coverage''',
                'topic': 'Tokenization',
                'type': 'concept'
            },
            {
                'front': 'What is the difference between precision and recall?',
                'back': '''<b>Precision:</b> TP/(TP+FP) - accuracy of positive predictions<br>
<b>Recall:</b> TP/(TP+FN) - fraction of positives found<br>
<b>Trade-off:</b> High precision → low recall, vice versa<br>
<b>F1:</b> Harmonic mean balances both metrics''',
                'topic': 'Evaluation',
                'type': 'concept'
            }
        ]
        
        for card_data in general_cards:
            card = self.create_card(
                front=card_data['front'],
                back=card_data['back'],
                topic=card_data['topic'],
                card_type=card_data['type']
            )
            all_cards.append(card)
        
        # Create deck structure
        deck_data = {
            "__type__": "Deck",
            "children": [],
            "crowdanki_uuid": "nlp-optimal-interview-deck-2024",
            "deck_config_uuid": "nlp-optimal-deck-config",
            "deck_configurations": [{
                "__type__": "DeckConfig",
                "autoplay": True,
                "crowdanki_uuid": "nlp-optimal-deck-config",
                "dyn": False,
                "name": "NLP Interview Flashcards (Optimal)",
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
            "desc": "Research-backed NLP interview flashcards. High-level concepts, formulas with definitions, 15-20 second review time.",
            "dyn": 0,
            "extendNew": 10,
            "extendRev": 50,
            "media_files": [],
            "name": "NLP Interview Flashcards (Optimal)",
            "note_models": [{
                "__type__": "NoteModel",
                "crowdanki_uuid": "nlp-optimal-model",
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
                "name": "NLP Optimal",
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
            "notes": all_cards
        }
        
        return deck_data

def main():
    """Generate optimal flashcard deck."""
    generator = OptimalAnkiGenerator()
    deck_data = generator.create_optimal_deck()
    
    # Save to NLP_Interview_Flashcards folder
    output_dir = Path('NLP_Interview_Flashcards')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'NLP_Interview_Flashcards.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deck_data, f, indent=2, ensure_ascii=False)
    
    total_cards = len(deck_data['notes'])
    
    print(f"\n✅ Optimal NLP Interview Flashcards Created!")
    print(f"📁 File: {output_file}")
    print(f"📊 Total Cards: {total_cards}")
    print(f"\n🎯 Features:")
    print(f"  ✓ High-level concepts (no code)")
    print(f"  ✓ Formulas with symbol definitions")
    print(f"  ✓ 2-4 line answers")
    print(f"  ✓ 15-20 second review time")
    print(f"  ✓ Interview-focused insights")
    print(f"  ✓ Mobile-optimized design")
    print(f"\n🚀 Ready for Anki import!")

if __name__ == "__main__":
    main()
