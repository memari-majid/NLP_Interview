#!/usr/bin/env python3
"""
Convert NLP problems/solutions to CrowdAnki JSON format.
Usage: python convert_to_anki.py
Output: anki_deck/NLP_Interview_Deck.json
"""

import json
import os
import re
import hashlib
from pathlib import Path

def clean_code_for_anki(code):
    """Clean Python code for Anki display."""
    # Remove excessive blank lines
    lines = code.split('\n')
    cleaned = []
    prev_blank = False
    for line in lines:
        if line.strip() == '':
            if not prev_blank:
                cleaned.append(line)
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return '\n'.join(cleaned).strip()

def extract_key_concepts(solution_content):
    """Extract key concepts from solution comments."""
    concepts = []
    # Look for "Key:" or "Interview:" or "Follow-up:" comments
    for line in solution_content.split('\n'):
        if any(marker in line for marker in ['Key:', 'Interview:', 'Follow-up:', 'Complexity:', 'Edge:']):
            concepts.append(line.strip().lstrip('#').strip())
    return '<br>'.join(concepts[:5])  # Top 5 key points

def markdown_to_html(text):
    """Simple markdown to HTML conversion for Anki."""
    # Convert code blocks
    text = re.sub(r'```python\n(.*?)\n```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'```\n(.*?)\n```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    # Convert inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Convert bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # Convert lists
    text = re.sub(r'^- (.+)$', r'• \1', text, flags=re.MULTILINE)
    # Convert newlines
    text = text.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return text

def create_note_id(content):
    """Create stable note ID from content hash."""
    return int(hashlib.md5(content.encode()).hexdigest()[:13], 16)

def convert_problem_to_anki(problem_path, solution_path, topic):
    """Convert a problem/solution pair to Anki note format."""
    # Read problem
    with open(problem_path, 'r') as f:
        problem_content = f.read()
    
    # Read solution
    with open(solution_path, 'r') as f:
        solution_content = f.read()
    
    # Extract title from problem
    title_match = re.search(r'^#\s*(.+)$', problem_content, re.MULTILINE)
    title = title_match.group(1) if title_match else os.path.basename(problem_path)
    
    # Clean problem description
    problem_desc = re.sub(r'^#.*$', '', problem_content, flags=re.MULTILINE).strip()
    
    # Extract core solution (first function/class)
    solution_match = re.search(r'(def\s+\w+.*?(?=\n\n|\Z))|(class\s+\w+.*?(?=\n\n|\Z))', 
                              solution_content, re.DOTALL)
    core_solution = solution_match.group(0) if solution_match else solution_content[:500]
    
    # Create different card types
    cards = []
    
    # Card 1: Problem → Implementation
    cards.append({
        "__type__": "Note",
        "fields": [
            f"<b>{title}</b><br><br>{markdown_to_html(problem_desc)}",  # Front
            f"<pre><code>{clean_code_for_anki(core_solution)}</code></pre>",  # Back
            topic,  # Tags
            "implementation"  # Card type
        ],
        "guid": f"nlp_{create_note_id(title + '_impl')}",
        "note_model_uuid": "nlp-model-basic",
        "tags": [topic.lower().replace(' ', '_'), "nlp_interview", "implementation"]
    })
    
    # Card 2: Concept → Key Points
    key_concepts = extract_key_concepts(solution_content)
    if key_concepts:
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{title}</b><br>What are the key concepts?",  # Front
                key_concepts,  # Back
                topic,  # Tags
                "concepts"  # Card type
            ],
            "guid": f"nlp_{create_note_id(title + '_concepts')}",
            "note_model_uuid": "nlp-model-basic",
            "tags": [topic.lower().replace(' ', '_'), "nlp_interview", "concepts"]
        })
    
    # Card 3: Quick recall formula/complexity
    complexity_match = re.search(r'(?:Time|Space).*?O\([^)]+\)', solution_content)
    if complexity_match:
        cards.append({
            "__type__": "Note",
            "fields": [
                f"<b>{title}</b><br>Time/Space Complexity?",  # Front
                complexity_match.group(0),  # Back
                topic,  # Tags
                "complexity"  # Card type
            ],
            "guid": f"nlp_{create_note_id(title + '_complexity')}",
            "note_model_uuid": "nlp-model-basic",
            "tags": [topic.lower().replace(' ', '_'), "nlp_interview", "complexity"]
        })
    
    return cards

def create_anki_deck():
    """Create complete CrowdAnki deck from all problems."""
    
    # Deck structure
    deck = {
        "__type__": "Deck",
        "children": [],  # Subdecks
        "crowdanki_uuid": "nlp-interview-deck-2024",
        "deck_config_uuid": "nlp-deck-config-1",
        "deck_configurations": [{
            "__type__": "DeckConfig",
            "autoplay": True,
            "crowdanki_uuid": "nlp-deck-config-1",
            "dyn": False,
            "name": "NLP Interview Settings",
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
        "desc": "NLP coding interview questions with implementations, concepts, and complexity analysis.",
        "dyn": 0,
        "extendNew": 10,
        "extendRev": 50,
        "media_files": [],
        "name": "NLP Interview Preparation",
        "note_models": [{
            "__type__": "NoteModel",
            "crowdanki_uuid": "nlp-model-basic",
            "css": """
.card {
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 16px;
    text-align: left;
    color: #333;
    background-color: #f5f5f5;
}
pre {
    background-color: #282c34;
    color: #abb2bf;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
}
code {
    background-color: #e1e4e8;
    padding: 2px 4px;
    border-radius: 3px;
    color: #d73a49;
}
b {
    color: #0366d6;
}
            """,
            "flds": [
                {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Back", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Topic", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
                {"name": "Type", "ord": 3, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""}
            ],
            "name": "NLP Interview Card",
            "tmpls": [{
                "afmt": "{{FrontSide}}<hr id=answer>{{Back}}<br><br><small>Topic: {{Topic}}</small>",
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
    topic_notes = {}
    
    for topic_dir in sorted(nlp_dir.iterdir()):
        if topic_dir.is_dir():
            topic_name = topic_dir.name.replace('_', ' ')
            topic_notes[topic_name] = []
            
            # Find problem/solution pairs
            problem_files = list(topic_dir.glob("*_problem.md"))
            
            for problem_file in problem_files:
                solution_file = problem_file.with_suffix('.py').with_name(
                    problem_file.stem.replace('_problem', '_solution') + '.py'
                )
                
                if solution_file.exists():
                    notes = convert_problem_to_anki(problem_file, solution_file, topic_name)
                    topic_notes[topic_name].extend(notes)
    
    # Create subdecks by topic
    for topic, notes in topic_notes.items():
        if notes:  # Only create subdeck if it has notes
            subdeck = {
                "__type__": "Deck",
                "children": [],
                "crowdanki_uuid": f"nlp-subdeck-{topic.lower().replace(' ', '-')}",
                "deck_config_uuid": "nlp-deck-config-1",
                "desc": f"NLP Interview: {topic}",
                "dyn": 0,
                "extendNew": 10,
                "extendRev": 50,
                "name": f"NLP Interview Preparation::{topic}",
                "notes": notes
            }
            deck["children"].append(subdeck)
    
    # Also add all notes to main deck for those who prefer flat structure
    all_notes = []
    for notes in topic_notes.values():
        all_notes.extend(notes)
    deck["notes"] = all_notes
    
    return deck

def main():
    """Generate CrowdAnki JSON file."""
    print("Converting NLP problems to Anki format...")
    
    # Create output directory
    output_dir = Path("anki_deck")
    output_dir.mkdir(exist_ok=True)
    
    # Generate deck
    deck = create_anki_deck()
    
    # Write JSON
    output_file = output_dir / "NLP_Interview_Deck.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deck, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Anki deck created: {output_file}")
    print(f"📊 Total cards: {sum(len(subdeck['notes']) for subdeck in deck['children'])}")
    print("\nTo import in Anki:")
    print("1. Install CrowdAnki addon (code: 1788670778)")
    print("2. File → CrowdAnki: Import from disk")
    print("3. Select the 'anki_deck' folder")
    print("\nDeck structure:")
    for subdeck in deck['children']:
        topic = subdeck['name'].split('::')[-1]
        count = len(subdeck['notes'])
        print(f"  - {topic}: {count} cards")

if __name__ == "__main__":
    main()
