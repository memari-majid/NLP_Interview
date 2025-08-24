#!/usr/bin/env python3
"""
Categorized NLP Interview Flashcard Generator
Creates separate Anki decks organized by topic for better study organization.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Set

class CategorizedAnkiGenerator:
    """Generate topic-specific Anki decks for organized studying."""
    
    def __init__(self):
        self.deck_categories = self._define_deck_categories()
        
    def _define_deck_categories(self) -> Dict:
        """Define the main deck categories and their cards."""
        return {
            'fundamentals': {
                'name': 'NLP Fundamentals',
                'description': 'Core concepts every NLP engineer should know',
                'cards': [
                    {
                        'front': 'What is the curse of dimensionality in NLP?',
                        'back': '''<b>Problem:</b> Sparse data in high-dimensional spaces<br>
<b>Effect:</b> Most vectors are nearly orthogonal<br>
<b>Solution:</b> Embeddings reduce to dense, lower dimensions<br>
<b>Example:</b> One-hot vectors vs 300d word embeddings'''
                    },
                    {
                        'front': 'Why do we need tokenization?',
                        'back': '''<b>Purpose:</b> Split text into meaningful units for processing<br>
<b>Challenges:</b> Punctuation, contractions, out-of-vocabulary words<br>
<b>Modern:</b> Subword tokenization (BPE, SentencePiece)<br>
<b>Benefit:</b> Balance between vocabulary size and coverage'''
                    },
                    {
                        'front': 'What is the difference between precision and recall?',
                        'back': '''<b>Precision:</b> TP/(TP+FP) - accuracy of positive predictions<br>
<b>Recall:</b> TP/(TP+FN) - fraction of positives found<br>
<b>Trade-off:</b> High precision → low recall, vice versa<br>
<b>F1:</b> Harmonic mean balances both metrics'''
                    },
                    {
                        'front': 'What are n-grams and why are they useful?',
                        'back': '''<b>Definition:</b> Contiguous sequences of n tokens<br>
<b>Examples:</b> Unigrams (words), bigrams (word pairs), trigrams<br>
<b>Use:</b> Language modeling, feature extraction<br>
<b>Trade-off:</b> Higher n captures more context but increases sparsity'''
                    }
                ]
            },
            'attention_transformers': {
                'name': 'Attention & Transformers',
                'description': 'Modern neural architectures powering current NLP',
                'cards': [
                    {
                        'front': 'What is self-attention?',
                        'back': '''<b>Intuition:</b> Weigh tokens by relevance to each other in sequence<br>
<b>Formula:</b> Attention(Q,K,V) = softmax(QK^T/√d_k)V<br>
<b>Symbols:</b> Q=query, K=key, V=value, d_k=key dimension<br>
<b>Why:</b> Enables parallel computation and long-range dependencies'''
                    },
                    {
                        'front': 'Write the attention formula with scaling',
                        'back': '''<b>Formula:</b> softmax(QK^T/√d_k)V<br>
<b>Symbols:</b> Q=query matrix, K=key matrix, V=value matrix<br>
<b>Scaling:</b> √d_k prevents vanishing gradients in softmax<br>
<b>Output:</b> Weighted sum of values based on attention scores'''
                    },
                    {
                        'front': 'Why use attention over RNNs?',
                        'back': '''<b>Parallelization:</b> All positions computed simultaneously<br>
<b>Long-range:</b> Direct connections between any two positions<br>
<b>Interpretability:</b> Attention weights show model focus<br>
<b>Performance:</b> Better at capturing dependencies in long sequences'''
                    },
                    {
                        'front': 'Why do transformers work so well?',
                        'back': '''<b>Parallelization:</b> No sequential bottleneck like RNNs<br>
<b>Attention:</b> Direct connections between all positions<br>
<b>Scalability:</b> Performance improves with more data/compute<br>
<b>Transfer:</b> Pre-trained models adapt well to new tasks'''
                    },
                    {
                        'front': 'Key components of transformer architecture?',
                        'back': '''<b>Multi-head attention:</b> Parallel attention mechanisms<br>
<b>Position encoding:</b> Inject sequence order information<br>
<b>Layer norm + residual:</b> Training stability<br>
<b>Feed-forward:</b> Non-linear transformations per position'''
                    }
                ]
            },
            'embeddings_similarity': {
                'name': 'Embeddings & Similarity',
                'description': 'Vector representations and measuring text similarity',
                'cards': [
                    {
                        'front': 'What are word embeddings?',
                        'back': '''<b>Intuition:</b> Map discrete tokens to dense continuous vectors<br>
<b>Property:</b> Similar words have similar vectors<br>
<b>Training:</b> Word2Vec (skip-gram/CBOW) or context prediction<br>
<b>Benefit:</b> Captures semantic relationships in vector space'''
                    },
                    {
                        'front': 'Word2Vec vs contextual embeddings?',
                        'back': '''<b>Word2Vec:</b> One vector per word type (fast, fixed)<br>
<b>Contextual:</b> Different vectors per word occurrence<br>
<b>Example:</b> "bank" (river) vs "bank" (money)<br>
<b>Trade-off:</b> Contextual more accurate but computationally expensive'''
                    },
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
                    }
                ]
            },
            'classical_methods': {
                'name': 'Classical NLP Methods',
                'description': 'Traditional approaches still used in production',
                'cards': [
                    {
                        'front': 'What is TF-IDF?',
                        'back': '''<b>Intuition:</b> Weight terms by frequency and rarity across documents<br>
<b>Formula:</b> TF-IDF = TF × log(N/df)<br>
<b>Symbols:</b> TF=term freq, N=total docs, df=docs with term<br>
<b>Use:</b> Information retrieval and document similarity'''
                    },
                    {
                        'front': 'Write the TF-IDF formula and explain components',
                        'back': '''<b>Formula:</b> TF(t,d) × IDF(t,D) = (count/|d|) × log(N/df)<br>
<b>TF:</b> Term frequency in document (normalized by doc length)<br>
<b>IDF:</b> Inverse document frequency (rarity measure)<br>
<b>Result:</b> High for frequent terms in few documents'''
                    },
                    {
                        'front': 'Common TF-IDF edge case and fix?',
                        'back': '''<b>Problem:</b> Division by zero when df=0<br>
<b>Fix:</b> Add smoothing: log(N/(df+1))<br>
<b>Alternative:</b> Filter out terms not in vocabulary<br>
<b>Impact:</b> Prevents runtime errors in production'''
                    },
                    {
                        'front': 'What are the gates in LSTM?',
                        'back': '''<b>Forget:</b> What to discard from cell state<br>
<b>Input:</b> What new information to store<br>
<b>Output:</b> What parts of cell state to output<br>
<b>Purpose:</b> Control information flow and prevent vanishing gradients'''
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
            "tags": [topic.lower().replace(' ', '_'), card_type]
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
                    "perDay": 15  # Smaller per-deck limit
                },
                "rev": {
                    "ease4": 1.3,
                    "hardFactor": 1.2,
                    "ivlFct": 1.0,
                    "maxIvl": 36500,
                    "perDay": 50  # Smaller review limit
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
    
    def generate_all_decks(self):
        """Generate all categorized decks."""
        output_base = Path('NLP_Interview_Decks')
        output_base.mkdir(exist_ok=True)
        
        # Clean up any existing files
        for file in output_base.glob('*.json'):
            file.unlink()
        
        total_cards = 0
        deck_summary = []
        
        for deck_key, deck_info in self.deck_categories.items():
            # Generate cards for this deck
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
            
            # Save deck file
            filename = f"{deck_key.title().replace('_', '')}_Flashcards.json"
            output_file = output_base / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(deck_data, f, indent=2, ensure_ascii=False)
            
            deck_summary.append({
                'name': deck_info['name'],
                'file': filename,
                'cards': len(cards),
                'description': deck_info['description']
            })
            total_cards += len(cards)
        
        # Create summary README
        self._create_summary_readme(output_base, deck_summary, total_cards)
        
        return deck_summary, total_cards
    
    def _create_summary_readme(self, output_dir: Path, deck_summary: List[Dict], total_cards: int):
        """Create a comprehensive README for all decks."""
        readme_content = f"""# 🎯 NLP Interview Flashcard Decks

## 📚 Overview
Organized flashcard decks for systematic NLP interview preparation. Each deck focuses on a specific area for targeted study.

## 📊 Deck Summary
**Total Cards**: {total_cards} across {len(deck_summary)} specialized decks

| Deck | Cards | Focus Area |
|------|-------|------------|
"""
        
        for deck in deck_summary:
            readme_content += f"| **{deck['name']}** | {deck['cards']} | {deck['description']} |\n"
        
        readme_content += f"""
## 🚀 Import Instructions

### Option 1: Import Individual Decks (Recommended)
1. **Choose your focus area** from the table above
2. **Copy the specific JSON file** to your computer
3. **Import to Anki** using CrowdAnki: `File` → `CrowdAnki: Import from disk`
4. **Select the JSON file** directly

### Option 2: Import All Decks
1. **Copy entire `NLP_Interview_Decks` folder** to your computer
2. **Import each JSON file separately** (Anki will create separate decks)

## 📱 Study Strategy

### Focused Learning Path
1. **Start with Fundamentals** (core concepts)
2. **Move to Classical Methods** (traditional NLP)
3. **Study Embeddings & Similarity** (vector representations)
4. **Master Attention & Transformers** (modern architectures)

### Daily Schedule Recommendation
- **New cards**: 10-15 per deck per day
- **Reviews**: 30-50 per deck per day
- **Focus**: Study 1-2 decks at a time for depth

## 🎯 Deck Details

"""
        
        for deck in deck_summary:
            readme_content += f"""### {deck['name']}
**File**: `{deck['file']}`  
**Cards**: {deck['cards']}  
**Focus**: {deck['description']}

"""
        
        readme_content += """## ✅ Quality Features
- **15-20 second review time** per card
- **High-level concepts** (no code memorization)
- **Complete formulas** with symbol definitions
- **Mobile-optimized** formatting
- **Research-backed** design principles

## 🔄 Progressive Learning
Start with any deck based on your current knowledge level. Each deck is self-contained but builds upon fundamental concepts.
"""
        
        readme_file = output_dir / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

def main():
    """Generate all categorized flashcard decks."""
    generator = CategorizedAnkiGenerator()
    deck_summary, total_cards = generator.generate_all_decks()
    
    print(f"\n✅ Categorized NLP Interview Flashcards Created!")
    print(f"📁 Location: NLP_Interview_Decks/")
    print(f"📊 Total: {total_cards} cards across {len(deck_summary)} decks")
    print(f"\n🎯 Decks Created:")
    
    for deck in deck_summary:
        print(f"  📚 {deck['name']}: {deck['cards']} cards")
        print(f"     File: {deck['file']}")
        print(f"     Focus: {deck['description']}")
        print()
    
    print(f"🚀 Import any deck individually for focused study!")
    print(f"📱 Each deck optimized for 15-20 minute study sessions")

if __name__ == "__main__":
    main()
