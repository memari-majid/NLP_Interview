#!/usr/bin/env python3
"""
Optimized Anki card generator that creates bite-sized cards.
Each function/concept becomes its own card for better memorization.
"""

import json
import os
import re
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

def extract_functions(code: str) -> List[Dict[str, str]]:
    """Extract individual functions with their docstrings and comments."""
    functions = []
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function source
                start_line = node.lineno - 1
                end_line = node.end_lineno
                
                # Extract function lines
                lines = code.split('\n')
                func_lines = lines[start_line:end_line]
                func_code = '\n'.join(func_lines)
                
                # Get docstring
                docstring = ast.get_docstring(node) or ""
                
                # Extract key concepts from comments
                key_concepts = []
                for line in func_lines:
                    if '#' in line and any(marker in line for marker in ['Key:', 'Interview:', 'Complexity:', 'Edge:']):
                        key_concepts.append(line.strip())
                
                functions.append({
                    'name': node.name,
                    'code': func_code,
                    'docstring': docstring,
                    'key_concepts': key_concepts,
                    'lines': len(func_lines)
                })
    except:
        # Fallback for non-parseable code
        pass
    
    return functions

def create_formula_cards(solution_content: str) -> List[Dict[str, str]]:
    """Extract mathematical formulas and create cards for them."""
    formulas = []
    
    # Common NLP formulas patterns
    patterns = [
        (r'IDF.*?=.*?(?:ln|log).*?\)', 'TF-IDF Formula'),
        (r'(?:cosine|similarity).*?=.*?/', 'Cosine Similarity'),
        (r'attention.*?=.*?(?:QK|softmax)', 'Attention Formula'),
        (r'(?:precision|recall|f1).*?=.*?/', 'Evaluation Metric'),
        (r'loss.*?=.*?(?:log|exp)', 'Loss Function'),
    ]
    
    for pattern, name in patterns:
        matches = re.findall(pattern, solution_content, re.IGNORECASE)
        for match in matches:
            formulas.append({
                'name': name,
                'formula': match.strip()
            })
    
    return formulas

def create_complexity_cards(solution_content: str) -> List[Dict[str, str]]:
    """Extract complexity analysis."""
    complexities = []
    
    # Find complexity mentions
    time_pattern = r'(?:Time|time).*?O\([^)]+\)'
    space_pattern = r'(?:Space|space).*?O\([^)]+\)'
    
    time_matches = re.findall(time_pattern, solution_content)
    space_matches = re.findall(space_pattern, solution_content)
    
    for match in time_matches:
        complexities.append({
            'type': 'Time Complexity',
            'value': match
        })
    
    for match in space_matches:
        complexities.append({
            'type': 'Space Complexity', 
            'value': match
        })
    
    return complexities

def create_edge_case_cards(solution_content: str) -> List[Dict[str, str]]:
    """Extract edge cases and error handling."""
    edge_cases = []
    
    # Find edge case patterns
    patterns = [
        r'if\s+not\s+\w+.*?:.*?(?:return|raise)',
        r'if\s+.*?(?:is None|== None|is_empty|len.*?== 0)',
        r'(?:# Edge|# Handle|# Check).*',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, solution_content, re.MULTILINE | re.DOTALL)
        for match in matches:
            if len(match) < 200:  # Keep it concise
                edge_cases.append(match.strip())
    
    return edge_cases

def chunk_code(code: str, max_lines: int = 15) -> List[str]:
    """Break code into smaller chunks for cards."""
    lines = code.strip().split('\n')
    chunks = []
    
    current_chunk = []
    current_size = 0
    
    for line in lines:
        # Start new chunk if current is too big
        if current_size >= max_lines and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(line)
        current_size += 1
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def create_optimized_anki_cards(problem_path: Path, solution_path: Path, topic: str) -> List[Dict]:
    """Create multiple focused cards from each problem/solution."""
    cards = []
    
    # Read files
    with open(problem_path, 'r') as f:
        problem_content = f.read()
    
    with open(solution_path, 'r') as f:
        solution_content = f.read()
    
    # Extract problem title
    title_match = re.search(r'^#\s*(.+)$', problem_content, re.MULTILINE)
    title = title_match.group(1) if title_match else problem_path.stem
    
    # Extract problem description
    problem_lines = problem_content.split('\n')
    problem_desc = []
    for line in problem_lines:
        if not line.startswith('#') and line.strip():
            problem_desc.append(line)
    problem_text = ' '.join(problem_desc[:3])  # First 3 lines
    
    # 1. Problem Understanding Card
    cards.append({
        "__type__": "Note",
        "fields": [
            f"<b>{title}</b><br>What is the key insight?",
            f"{problem_text[:150]}...<br><br>Think: What algorithm/approach?",
            topic,
            "problem_understanding"
        ],
        "guid": f"nlp_{hashlib.md5(f'{title}_understanding'.encode()).hexdigest()[:13]}",
        "note_model_uuid": "nlp-model-basic",
        "tags": [topic.lower().replace(' ', '_'), "understanding"]
    })
    
    # 2. Function Cards - One per function
    functions = extract_functions(solution_content)
    for func in functions[:5]:  # Limit to 5 main functions
        if func['lines'] <= 20:  # Only include concise functions
            cards.append({
                "__type__": "Note",
                "fields": [
                    f"<b>{title}</b><br>Implement: {func['name']}()",
                    f"<pre><code>{func['code']}</code></pre>",
                    topic,
                    "implementation"
                ],
                "guid": f"nlp_{hashlib.md5(f'{title}_{func["name"]}'.encode()).hexdigest()[:13]}",
                "note_model_uuid": "nlp-model-basic",
                "tags": [topic.lower().replace(' ', '_'), "function"]
            })
    
    # 3. Formula Cards
    formulas = create_formula_cards(solution_content)
    for formula in formulas[:3]:  # Top 3 formulas
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{title}</b><br>Write the {formula['name']}",
                formula['formula'],
                topic,
                "formula"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_{formula["name"]}'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-basic",
            "tags": [topic.lower().replace(' ', '_'), "formula"]
        })
    
    # 4. Complexity Card
    complexities = create_complexity_cards(solution_content)
    if complexities:
        complexity_text = '<br>'.join([f"{c['type']}: {c['value']}" for c in complexities])
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{title}</b><br>What's the complexity?",
                complexity_text,
                topic,
                "complexity"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_complexity'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-basic",
            "tags": [topic.lower().replace(' ', '_'), "complexity"]
        })
    
    # 5. Edge Case Cards
    edge_cases = create_edge_case_cards(solution_content)
    if edge_cases:
        edge_text = '<br>'.join(edge_cases[:3])  # Top 3 edge cases
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{title}</b><br>What edge cases to handle?",
                f"<code>{edge_text}</code>",
                topic,
                "edge_cases"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_edges'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-basic",
            "tags": [topic.lower().replace(' ', '_'), "edge_cases"]
        })
    
    # 6. Key Insight Card (from comments)
    key_insights = re.findall(r'#\s*(?:Key|Interview|Important):\s*(.+)', solution_content)
    if key_insights:
        insights_text = '<br>'.join(key_insights[:3])
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{title}</b><br>Interview talking points?",
                insights_text,
                topic,
                "insights"
            ],
            "guid": f"nlp_{hashlib.md5(f'{title}_insights'.encode()).hexdigest()[:13]}",
            "note_model_uuid": "nlp-model-basic",
            "tags": [topic.lower().replace(' ', '_'), "interview_tips"]
        })
    
    return cards

def create_anki_deck_optimized():
    """Create optimized Anki deck with bite-sized cards."""
    
    # Deck structure with custom CSS for mobile
    deck = {
        "__type__": "Deck",
        "children": [],
        "crowdanki_uuid": "nlp-interview-deck-optimized-2024",
        "deck_config_uuid": "nlp-deck-config-optimized",
        "deck_configurations": [{
            "__type__": "DeckConfig",
            "autoplay": True,
            "crowdanki_uuid": "nlp-deck-config-optimized",
            "dyn": False,
            "name": "NLP Interview Optimized",
            "new": {
                "delays": [1, 10],
                "initialFactor": 2500,
                "ints": [1, 4, 7],
                "order": 1,
                "perDay": 30  # More cards since they're smaller
            },
            "rev": {
                "ease4": 1.3,
                "hardFactor": 1.2,
                "ivlFct": 1.0,
                "maxIvl": 36500,
                "perDay": 100
            }
        }],
        "desc": "Bite-sized NLP interview cards optimized for mobile learning.",
        "dyn": 0,
        "extendNew": 10,
        "extendRev": 50,
        "media_files": [],
        "name": "NLP Interview Prep (Optimized)",
        "note_models": [{
            "__type__": "NoteModel",
            "crowdanki_uuid": "nlp-model-basic",
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
    color: #333;
    background-color: #fff;
    padding: 15px;
    max-width: 600px;
    margin: 0 auto;
}
pre {
    background-color: #f6f8fa;
    color: #24292e;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 14px;
    line-height: 1.45;
    font-family: 'SF Mono', Monaco, Consolas, monospace;
}
code {
    background-color: #f3f4f6;
    padding: 2px 6px;
    border-radius: 3px;
    color: #e01e5a;
    font-size: 14px;
}
b {
    color: #0969da;
    font-weight: 600;
}
/* Mobile optimizations */
@media (max-width: 600px) {
    .card { font-size: 15px; padding: 10px; }
    pre { font-size: 12px; padding: 8px; }
    code { font-size: 13px; }
}
            """,
            "flds": [
                {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Back", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Topic", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Type", "ord": 3, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""}
            ],
            "name": "NLP Interview Card (Optimized)",
            "tmpls": [{
                "afmt": "{{FrontSide}}<hr id=answer>{{Back}}<br><br><small style='color:#666'>{{Topic}} • {{Type}}</small>",
                "bqfmt": "",
                "did": None,
                "name": "Card 1",
                "ord": 0,
                "qfmt": "{{Front}}"
            }],
            "type": 0
        }],
        "notes": []
    }
    
    # Process all problems
    nlp_dir = Path("NLP")
    all_notes = []
    
    for topic_dir in sorted(nlp_dir.iterdir()):
        if topic_dir.is_dir():
            topic_name = topic_dir.name.replace('_', ' ')
            
            # Find problem/solution pairs
            problem_files = list(topic_dir.glob("*_problem.md"))
            
            for problem_file in problem_files:
                solution_file = problem_file.with_suffix('.py').with_name(
                    problem_file.stem.replace('_problem', '_solution') + '.py'
                )
                
                if solution_file.exists():
                    notes = create_optimized_anki_cards(problem_file, solution_file, topic_name)
                    all_notes.extend(notes)
    
    deck["notes"] = all_notes
    return deck

def main():
    """Generate optimized Anki deck."""
    print("Generating optimized Anki deck with bite-sized cards...")
    
    # Create output directory
    output_dir = Path("anki_deck_optimized")
    output_dir.mkdir(exist_ok=True)
    
    # Generate deck
    deck = create_anki_deck_optimized()
    
    # Write JSON
    output_file = output_dir / "NLP_Interview_Optimized.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deck, f, ensure_ascii=False, indent=2)
    
    # Count cards by type
    card_types = {}
    for note in deck["notes"]:
        card_type = note["fields"][3]
        card_types[card_type] = card_types.get(card_type, 0) + 1
    
    print(f"✅ Optimized deck created: {output_file}")
    print(f"📊 Total cards: {len(deck['notes'])}")
    print("\nCard breakdown:")
    for card_type, count in sorted(card_types.items()):
        print(f"  - {card_type}: {count} cards")
    
    print("\n💡 Optimization features:")
    print("  - Bite-sized cards (max 15-20 lines)")
    print("  - One concept per card")
    print("  - Mobile-optimized CSS")
    print("  - 6 card types per problem")
    print("  - Focus on memorizable chunks")

if __name__ == "__main__":
    main()
