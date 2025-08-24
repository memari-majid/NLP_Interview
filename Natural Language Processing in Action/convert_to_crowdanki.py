#!/usr/bin/env python3
"""
Convert custom JSON flashcard format to CrowdAnki format for Anki import.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any

def create_note_model() -> Dict[str, Any]:
    """Create the note model (note type) definition for CrowdAnki."""
    return {
        "crowdanki_uuid": "nlp-comprehensive-note-model",
        "css": ".card {\n font-family: arial;\n font-size: 20px;\n text-align: center;\n color: black;\n background-color: white;\n}\n\n.front {\n font-weight: bold;\n color: #2c3e50;\n}\n\n.back {\n text-align: left;\n padding: 20px;\n}\n\n.concept {\n font-weight: bold;\n color: #e74c3c;\n margin-bottom: 10px;\n}\n\n.intuition {\n color: #3498db;\n font-style: italic;\n margin-bottom: 10px;\n}\n\n.mechanics {\n color: #27ae60;\n margin-bottom: 10px;\n}\n\n.tradeoffs {\n color: #f39c12;\n margin-bottom: 10px;\n}\n\n.applications {\n color: #9b59b6;\n margin-bottom: 10px;\n}\n\n.memory-hook {\n background-color: #ecf0f1;\n padding: 10px;\n border-left: 4px solid #34495e;\n font-style: italic;\n color: #34495e;\n}",
        "flds": [
            {
                "font": "Arial",
                "media": [],
                "name": "Front",
                "ord": 0,
                "rtl": False,
                "size": 20,
                "sticky": False
            },
            {
                "font": "Arial", 
                "media": [],
                "name": "Back",
                "ord": 1,
                "rtl": False,
                "size": 20,
                "sticky": False
            },
            {
                "font": "Arial",
                "media": [],
                "name": "Tags",
                "ord": 2,
                "rtl": False,
                "size": 20,
                "sticky": False
            },
            {
                "font": "Arial",
                "media": [],
                "name": "Difficulty", 
                "ord": 3,
                "rtl": False,
                "size": 20,
                "sticky": False
            }
        ],
        "latexPost": "\\end{document}",
        "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
        "name": "NLP Comprehensive",
        "req": [
            [
                0,
                "all"
            ]
        ],
        "sortf": 0,
        "tags": [],
        "tmpls": [
            {
                "afmt": "{{FrontSide}}\n\n<hr id=answer>\n\n<div class=\"back\">\n{{Back}}\n</div>",
                "bafmt": "",
                "bqfmt": "",
                "did": None,
                "name": "Card 1",
                "ord": 0,
                "qfmt": "<div class=\"front\">{{Front}}</div>"
            }
        ],
        "type": 0
    }

def format_answer(answer: Dict[str, str]) -> str:
    """Format the structured answer into HTML for the back of the card."""
    html_parts = []
    
    if 'concept' in answer:
        html_parts.append(f'<div class="concept"><strong>Concept:</strong> {answer["concept"]}</div>')
    
    if 'intuition' in answer:
        html_parts.append(f'<div class="intuition"><strong>Intuition:</strong> {answer["intuition"]}</div>')
        
    if 'mechanics' in answer:
        html_parts.append(f'<div class="mechanics"><strong>Mechanics:</strong> {answer["mechanics"]}</div>')
        
    if 'tradeoffs' in answer:
        html_parts.append(f'<div class="tradeoffs"><strong>Trade-offs:</strong> {answer["tradeoffs"]}</div>')
        
    if 'applications' in answer:
        html_parts.append(f'<div class="applications"><strong>Applications:</strong> {answer["applications"]}</div>')
        
    if 'memory_hook' in answer:
        html_parts.append(f'<div class="memory-hook"><strong>Memory Hook:</strong> {answer["memory_hook"]}</div>')
    
    return '<br><br>'.join(html_parts)

def convert_qa_to_note(qa_pair: Dict[str, Any], deck_id: int) -> Dict[str, Any]:
    """Convert a single QA pair to CrowdAnki note format."""
    # Create formatted answer
    if isinstance(qa_pair['answer'], dict):
        formatted_answer = format_answer(qa_pair['answer'])
    else:
        formatted_answer = str(qa_pair['answer'])
    
    # Join tags
    tags = qa_pair.get('tags', [])
    tags_str = ' '.join(tags)
    
    # Create note
    note = {
        "crowdanki_uuid": f"note-{hash(qa_pair['question'])}-{deck_id}",
        "fields": [
            qa_pair['question'],  # Front
            formatted_answer,     # Back
            tags_str,            # Tags
            qa_pair.get('difficulty', 'Medium')  # Difficulty
        ],
        "flags": 0,
        "guid": f"guid-{hash(qa_pair['question'])}-{deck_id}",
        "note_model_uuid": "nlp-comprehensive-note-model",
        "tags": tags
    }
    
    return note

def convert_json_to_crowdanki(input_file: str, output_file: str) -> None:
    """Convert a single JSON file from custom format to CrowdAnki format."""
    
    # Load the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract chapter name for deck name
    chapter_title = data.get('chapter', 'NLP Chapter')
    qa_pairs = data.get('qa_pairs', [])
    
    # Create deck ID (simple hash of chapter name)
    deck_id = abs(hash(chapter_title)) % 10000
    
    # Convert QA pairs to notes
    notes = []
    for qa_pair in qa_pairs:
        note = convert_qa_to_note(qa_pair, deck_id)
        notes.append(note)
    
    # Create CrowdAnki format
    crowdanki_data = {
        "crowdanki_uuid": f"deck-{deck_id}",
        "deck_config_uuid": "default-config",
        "deck_configurations": [
            {
                "crowdanki_uuid": "default-config",
                "name": "Default",
                "autoplay": True,
                "dyn": False,
                "lapse": {
                    "delays": [10],
                    "leechAction": 0,
                    "leechFails": 8,
                    "minInt": 1,
                    "mult": 0
                },
                "maxTaken": 60,
                "new": {
                    "bury": False,
                    "delays": [1, 10],
                    "initialFactor": 2500,
                    "ints": [1, 4, 0],
                    "order": 1,
                    "perDay": 20
                },
                "replayq": True,
                "rev": {
                    "bury": False,
                    "ease4": 1.3,
                    "hardFactor": 1.2,
                    "ivlFct": 1,
                    "maxIvl": 36500,
                    "perDay": 200
                },
                "timer": 0
            }
        ],
        "desc": f"Comprehensive flashcards for {chapter_title}",
        "dyn": False,
        "extendNew": 10,
        "extendRev": 50,
        "media_files": [],
        "name": chapter_title,
        "note_models": [create_note_model()],
        "notes": notes
    }
    
    # Write the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(crowdanki_data, f, indent=2, ensure_ascii=False)

def main():
    """Convert all JSON files in chapter directories."""
    base_path = Path('.')
    
    # Find all chapter directories
    chapter_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_', '11_', '12_'))]
    
    for chapter_dir in sorted(chapter_dirs):
        json_files = list(chapter_dir.glob('*.json'))
        for json_file in json_files:
            print(f"Converting {json_file}...")
            
            # Skip if already converted (check for note_models key)
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if 'note_models' in existing_data:
                        print(f"  Skipping {json_file} (already in CrowdAnki format)")
                        continue
            except:
                pass
            
            # Convert the file
            convert_json_to_crowdanki(str(json_file), str(json_file))
            print(f"  ✓ Converted {json_file}")

if __name__ == "__main__":
    main()
