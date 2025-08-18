#!/usr/bin/env python3
"""
Improved Anki card generator with complete, memorable solutions.
Creates focused Q&A pairs optimized for interview preparation.
"""

import json
import os
import re
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

def extract_core_implementation(code: str, func_name: str) -> str:
    """Extract the core implementation logic without excessive comments."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                lines = code.split('\n')[start_line:end_line]
                
                # Clean up excessive comments but keep key ones
                cleaned_lines = []
                for line in lines:
                    if line.strip().startswith('#'):
                        # Keep only important comments
                        if any(keyword in line.upper() for keyword in ['KEY:', 'FORMULA:', 'O(', 'COMPLEXITY']):
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
                
                return '\n'.join(cleaned_lines)
    except:
        pass
    return ""

def create_concept_cards(problem_path: Path, solution_path: Path, topic: str) -> List[Dict]:
    """Create focused concept cards for memorization."""
    cards = []
    
    with open(problem_path, 'r') as f:
        problem_content = f.read()
    
    with open(solution_path, 'r') as f:
        solution_content = f.read()
    
    # Extract problem title
    title_match = re.search(r'^#\s*(.+)$', problem_content, re.MULTILINE)
    title = title_match.group(1) if title_match else problem_path.stem.replace('_problem', '')
    
    # Clean title
    title = title.replace(' Problem', '').replace(' Implementation', '')
    
    # Extract key algorithm name
    algo_name = title.split('-')[0].strip() if '-' in title else title
    
    # 1. Core Algorithm Card - Complete implementation
    main_func_pattern = r'def\s+(\w+)\([^)]*\)[^:]*:\s*"""([^"]*)"""'
    main_funcs = re.findall(main_func_pattern, solution_content, re.DOTALL)
    
    if main_funcs:
        func_name, docstring = main_funcs[0]
        implementation = extract_core_implementation(solution_content, func_name)
        
        if implementation and len(implementation.split('\n')) <= 30:
            cards.append({
                "__type__": "Note",
                "fields": [
                    f"<b>{algo_name}</b><br>Write the complete implementation",
                    f"<pre><code>{implementation}</code></pre>",
                    topic,
                    "full_implementation"
                ],
                "guid": f"nlp_{hashlib.md5(f'{title}_full_impl'.encode()).hexdigest()[:13]}",
                "note_model_uuid": "nlp-model-improved",
                "tags": [topic.lower().replace(' ', '_'), "implementation", "core"]
            })
    
    # 2. Key Formula Card with Full Explanation
    if 'tfidf' in solution_path.name.lower():
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>TF-IDF</b><br>Write the formula and explain each part",
                "<pre>TF-IDF(t,d) = TF(t,d) × IDF(t)\n\nTF(t,d) = count(t in d) / total_terms(d)\nIDF(t) = log(N / df(t))\n\nWhere:\n- t = term\n- d = document\n- N = total documents\n- df(t) = documents containing t</pre>",
                topic,
                "formula"
            ],
            "guid": f"nlp_{hashlib.md5('tfidf_formula_complete'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-improved",
            "tags": ["tfidf", "formula"]
        })
    
    elif 'attention' in solution_path.name.lower():
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>Self-Attention</b><br>Write the attention formula",
                "<pre>Attention(Q,K,V) = softmax(QK^T / √d_k)V\n\nSteps:\n1. Compute scores: QK^T\n2. Scale: divide by √d_k\n3. Apply softmax for weights\n4. Multiply by V for output</pre>",
                topic,
                "formula"
            ],
            "guid": f"nlp_{hashlib.md5('attention_formula_complete'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-improved",
            "tags": ["attention", "formula"]
        })
    
    elif 'cosine' in solution_path.name.lower() or 'similarity' in solution_path.name.lower():
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>Cosine Similarity</b><br>Write the formula and implementation",
                "<pre>cosine_sim(A,B) = A·B / (||A|| × ||B||)\n\ndef cosine_similarity(vec1, vec2):\n    dot_product = sum(a*b for a,b in zip(vec1, vec2))\n    norm1 = sqrt(sum(a**2 for a in vec1))\n    norm2 = sqrt(sum(b**2 for b in vec2))\n    return dot_product / (norm1 * norm2)</pre>",
                topic,
                "formula"
            ],
            "guid": f"nlp_{hashlib.md5('cosine_formula_complete'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-improved",
            "tags": ["similarity", "formula"]
        })
    
    # 3. Step-by-Step Algorithm Card
    step_pattern = r'#\s*STEP\s*\d+:?\s*(.+)'
    steps = re.findall(step_pattern, solution_content)
    
    if steps:
        step_text = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(steps[:6])])
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{algo_name}</b><br>List the algorithm steps",
                f"<pre>{step_text}</pre>",
                topic,
                "algorithm_steps"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_steps'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-improved",
            "tags": [topic.lower().replace(' ', '_'), "steps"]
        })
    
    # 4. Complexity Analysis Card
    time_complexity = re.search(r'(?:Time|TIME)[^:]*:\s*O\([^)]+\)', solution_content, re.IGNORECASE)
    space_complexity = re.search(r'(?:Space|SPACE)[^:]*:\s*O\([^)]+\)', solution_content, re.IGNORECASE)
    
    if time_complexity or space_complexity:
        complexity_text = ""
        if time_complexity:
            complexity_text += time_complexity.group(0)
        if space_complexity:
            complexity_text += "\n" + space_complexity.group(0) if complexity_text else space_complexity.group(0)
        
        # Add explanation
        if 'tfidf' in solution_path.name.lower():
            complexity_text += "\n\nWhere:\n- d = number of documents\n- v = vocabulary size\n- n = average document length"
        elif 'attention' in solution_path.name.lower():
            complexity_text += "\n\nWhere:\n- n = sequence length\n- d = dimension size"
            
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{algo_name}</b><br>What's the time and space complexity?",
                f"<pre>{complexity_text}</pre>",
                topic,
                "complexity"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_complexity'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-improved",
            "tags": [topic.lower().replace(' ', '_'), "complexity"]
        })
    
    # 5. Key Insight Card
    key_insights = []
    
    # Look for explicit key points
    key_pattern = r'(?:Key|KEY|Interview|INTERVIEW)[^:]*:\s*(.+)'
    explicit_keys = re.findall(key_pattern, solution_content)
    key_insights.extend(explicit_keys[:3])
    
    # Add problem-specific insights
    if 'tfidf' in solution_path.name.lower():
        key_insights.append("TF-IDF balances term frequency with document rarity")
        key_insights.append("Use log in IDF to dampen the effect of very rare terms")
    elif 'attention' in solution_path.name.lower():
        key_insights.append("Divide by √d_k to prevent gradient vanishing in softmax")
        key_insights.append("Self-attention allows each position to attend to all positions")
    elif 'tokeniz' in solution_path.name.lower():
        key_insights.append("Handle punctuation, contractions, and special characters")
        key_insights.append("Consider subword tokenization (BPE) for OOV handling")
    
    if key_insights:
        insight_text = '\n'.join([f"• {insight}" for insight in key_insights[:4]])
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{algo_name}</b><br>What are the key interview talking points?",
                f"<pre>{insight_text}</pre>",
                topic,
                "insights"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_insights'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-improved",
            "tags": [topic.lower().replace(' ', '_'), "interview_tips"]
        })
    
    # 6. Common Mistakes Card
    mistakes = []
    if 'tfidf' in solution_path.name.lower():
        mistakes = [
            "Forgetting to handle empty documents",
            "Not normalizing TF by document length",
            "Using linear scale instead of log for IDF",
            "Division by zero in cosine similarity"
        ]
    elif 'attention' in solution_path.name.lower():
        mistakes = [
            "Forgetting to scale by √d_k",
            "Wrong dimension in matrix multiplication",
            "Not applying causal mask for autoregressive",
            "Incorrect softmax axis"
        ]
    
    if mistakes:
        mistakes_text = '\n'.join([f"❌ {m}" for m in mistakes])
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{algo_name}</b><br>What are common implementation mistakes?",
                f"<pre>{mistakes_text}</pre>",
                topic,
                "pitfalls"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_mistakes'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-improved",
            "tags": [topic.lower().replace(' ', '_'), "mistakes"]
        })
    
    return cards

def create_improved_anki_deck():
    """Create improved Anki deck with complete solutions."""
    
    deck = {
        "__type__": "Deck",
        "children": [],
        "crowdanki_uuid": "nlp-interview-deck-improved-2024",
        "deck_config_uuid": "nlp-deck-config-improved",
        "deck_configurations": [{
            "__type__": "DeckConfig",
            "autoplay": True,
            "crowdanki_uuid": "nlp-deck-config-improved",
            "dyn": False,
            "name": "NLP Interview Improved",
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
        "desc": "Complete NLP interview solutions optimized for memorization.",
        "dyn": 0,
        "extendNew": 10,
        "extendRev": 50,
        "media_files": [],
        "name": "NLP Interview Prep (Improved)",
        "note_models": [{
            "__type__": "NoteModel",
            "crowdanki_uuid": "nlp-model-improved",
            "sortf": 0,
            "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
            "latexPost": "\\end{document}",
            "tags": [],
            "vers": [],
            "css": """
.card {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 16px;
    text-align: left;
    color: #2d3748;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    min-height: 100vh;
}
.card-content {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    max-width: 700px;
    margin: 0 auto;
}
pre {
    background-color: #f7fafc;
    color: #2d3748;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 14px;
    line-height: 1.6;
    font-family: 'SF Mono', Monaco, Consolas, monospace;
    border-left: 4px solid #667eea;
}
code {
    background-color: #edf2f7;
    padding: 2px 6px;
    border-radius: 4px;
    color: #d6336c;
    font-size: 14px;
}
b {
    color: #5a67d8;
    font-weight: 600;
    font-size: 18px;
}
hr {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 20px 0;
}
/* Mobile optimizations */
@media (max-width: 600px) {
    .card { padding: 15px; }
    .card-content { padding: 15px; }
    pre { font-size: 12px; padding: 12px; }
}
            """,
            "flds": [
                {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Back", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Topic", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Type", "ord": 3, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""}
            ],
            "name": "NLP Interview Card (Improved)",
            "tmpls": [{
                "afmt": "<div class='card-content'>{{FrontSide}}<hr id=answer>{{Back}}<br><br><small style='color:#718096'>{{Topic}} • {{Type}}</small></div>",
                "bqfmt": "",
                "did": None,
                "name": "Card 1",
                "ord": 0,
                "qfmt": "<div class='card-content'>{{Front}}</div>"
            }],
            "type": 0
        }],
        "notes": []
    }
    
    # Process problems
    nlp_dir = Path("NLP")
    all_notes = []
    
    # Focus on key problems first
    priority_problems = [
        "tfidf", "attention", "tokenization", "word2vec", 
        "bert", "similarity", "classification", "bpe"
    ]
    
    for topic_dir in sorted(nlp_dir.iterdir()):
        if topic_dir.is_dir():
            topic_name = topic_dir.name.replace('_', ' ')
            
            # Check if this is a priority problem
            is_priority = any(p in topic_dir.name.lower() for p in priority_problems)
            
            problem_files = list(topic_dir.glob("*_problem.md"))
            
            for problem_file in problem_files:
                solution_file = problem_file.with_suffix('.py').with_name(
                    problem_file.stem.replace('_problem', '_solution') + '.py'
                )
                
                if solution_file.exists():
                    try:
                        notes = create_concept_cards(problem_file, solution_file, topic_name)
                        # Add more cards for priority problems
                        if is_priority:
                            all_notes.extend(notes)
                        else:
                            all_notes.extend(notes[:3])  # Fewer cards for non-priority
                    except Exception as e:
                        print(f"Error processing {problem_file}: {e}")
    
    deck["notes"] = all_notes
    return deck

def main():
    """Generate improved Anki deck."""
    print("Generating improved Anki deck with complete solutions...")
    
    output_dir = Path("anki_deck_improved")
    output_dir.mkdir(exist_ok=True)
    
    deck = create_improved_anki_deck()
    
    output_file = output_dir / "NLP_Interview_Improved.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deck, f, ensure_ascii=False, indent=2)
    
    # Stats
    card_types = {}
    for note in deck["notes"]:
        card_type = note["fields"][3]
        card_types[card_type] = card_types.get(card_type, 0) + 1
    
    print(f"✅ Improved deck created: {output_file}")
    print(f"📊 Total cards: {len(deck['notes'])}")
    print("\nCard breakdown:")
    for card_type, count in sorted(card_types.items()):
        print(f"  - {card_type}: {count} cards")
    
    print("\n🎯 Improvements:")
    print("  - Complete implementations (not fragments)")
    print("  - Full formulas with explanations")
    print("  - Step-by-step algorithms")
    print("  - Common mistakes to avoid")
    print("  - Enhanced visual design")

if __name__ == "__main__":
    main()