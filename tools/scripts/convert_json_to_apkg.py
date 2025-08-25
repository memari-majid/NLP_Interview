#!/usr/bin/env python3
"""
Convert JSON flashcard decks to APKG format using genanki library
"""

import json
import genanki
import os
from pathlib import Path
import random

# Create a custom note model for our flashcards
def create_note_model():
    """Create genanki note model matching our JSON structure"""
    return genanki.Model(
        model_id=random.randrange(1 << 30, 1 << 31),  # Random model ID
        name='NLP Interview Flashcard',
        fields=[
            {'name': 'Front'},
            {'name': 'Back'},
            {'name': 'Tags'},
            {'name': 'Difficulty'}
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '''
                    <div class="card">
                        <div class="front">{{Front}}</div>
                    </div>
                ''',
                'afmt': '''
                    {{FrontSide}}
                    
                    <hr id="answer">
                    
                    <div class="back">
                        {{Back}}
                    </div>
                    
                    <div class="difficulty">
                        <small>Difficulty: {{Difficulty}}</small>
                    </div>
                ''',
            },
        ],
        css='''
            .card {
                font-family: arial;
                font-size: 20px;
                text-align: center;
                color: black;
                background-color: white;
            }

            .front {
                font-weight: bold;
                color: #2c3e50;
                padding: 20px;
            }

            .back {
                text-align: left;
                padding: 20px;
                max-width: 800px;
                margin: 0 auto;
            }

            .concept {
                font-weight: bold;
                color: #e74c3c;
                margin-bottom: 10px;
            }

            .intuition {
                color: #3498db;
                font-style: italic;
                margin-bottom: 10px;
            }

            .mechanics {
                color: #27ae60;
                margin-bottom: 10px;
            }

            .tradeoffs {
                color: #f39c12;
                margin-bottom: 10px;
            }

            .applications {
                color: #9b59b6;
                margin-bottom: 10px;
            }

            .memory-hook {
                background-color: #ecf0f1;
                padding: 10px;
                border-left: 4px solid #34495e;
                font-style: italic;
                color: #34495e;
                margin-top: 10px;
            }

            .difficulty {
                text-align: center;
                color: #7f8c8d;
                font-size: 14px;
                margin-top: 15px;
            }
        '''
    )

def convert_json_to_deck(json_file_path, note_model):
    """Convert a single JSON file to genanki deck with validation"""
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error in {json_file_path}")
        print(f"   Line {e.lineno}, Column {e.colno}: {e.msg}")
        print(f"   Skipping this file...")
        return None
    except Exception as e:
        print(f"❌ Error reading {json_file_path}: {e}")
        return None
    
    deck_name = data.get('name', 'NLP Deck')
    notes_data = data.get('notes', [])
    
    # Create deck with hierarchical name preserved
    deck = genanki.Deck(
        deck_id=random.randrange(1 << 30, 1 << 31),
        name=deck_name
    )
    
    print(f"Converting {deck_name}...")
    print(f"  └── Cards: {len(notes_data)}")
    
    for note_data in notes_data:
        fields = note_data.get('fields', [])
        tags = note_data.get('tags', [])
        
        # Clean tags - remove spaces and invalid characters for Anki compatibility
        clean_tags = []
        for tag in tags:
            clean_tag = str(tag).replace(' ', '_').replace('-', '_')
            clean_tags.append(clean_tag)
        
        if len(fields) >= 4:
            note = genanki.Note(
                model=note_model,
                fields=fields,  # [Front, Back, Tags, Difficulty]
                tags=clean_tags
            )
            deck.add_note(note)
    
    return deck

def create_individual_apkg_files():
    """Create individual APKG files for each chapter"""
    base_path = Path('../../data/source/flashcards/NLP in Action')
    output_path = Path('../../data/output/apkg_files')
    output_path.mkdir(exist_ok=True)
    
    if not base_path.exists():
        print(f"❌ Path {base_path} not found!")
        return
    
    note_model = create_note_model()
    
    print("🔄 Converting JSON decks to APKG format...")
    print("=" * 60)
    
    # Get all chapter directories
    chapter_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_', '11_', '12_'))])
    
    created_files = []
    
    for chapter_dir in chapter_dirs:
        json_files = list(chapter_dir.glob("*.json"))
        if not json_files:
            print(f"❌ No JSON file in {chapter_dir.name}")
            continue
        
        json_file = json_files[0]
        deck = convert_json_to_deck(json_file, note_model)
        
        if deck is None:
            continue
        
        # Create clean filename
        safe_name = chapter_dir.name.replace('_', '-')
        output_file = output_path / f"{safe_name}.apkg"
        
        # Generate APKG file
        genanki.Package(deck).write_to_file(str(output_file))
        created_files.append(output_file)
        print(f"  ✅ Created: {output_file.name}")
    
    print("=" * 60)
    print(f"🎯 Success! Created {len(created_files)} APKG files")
    print(f"📁 Location: {output_path.absolute()}")
    print("\n📋 Files created:")
    for file in created_files:
        print(f"   • {file.name}")
    
    print(f"\n🚀 To import into Anki:")
    print(f"   1. Open Anki")
    print(f"   2. File → Import (or double-click APKG files)")
    print(f"   3. Select files from {output_path.absolute()}")
    print(f"   4. All hierarchical names (ML:NLP:XX) will be preserved!")

def create_master_apkg():
    """Create a single APKG file containing all chapters"""
    base_path = Path('../../data/source/flashcards/NLP in Action')
    output_path = Path('../../data/output/apkg_files')
    output_path.mkdir(exist_ok=True)
    
    if not base_path.exists():
        print(f"❌ Path {base_path} not found!")
        return
    
    note_model = create_note_model()
    
    # Create master deck
    master_deck = genanki.Deck(
        deck_id=random.randrange(1 << 30, 1 << 31),
        name="ML:NLP:Complete Collection"
    )
    
    print("🔄 Creating master APKG with all chapters...")
    print("=" * 60)
    
    chapter_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_', '11_', '12_'))])
    
    total_cards = 0
    
    for chapter_dir in chapter_dirs:
        json_files = list(chapter_dir.glob("*.json"))
        if not json_files:
            continue
        
        json_file = json_files[0]
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        notes_data = data.get('notes', [])
        chapter_name = data.get('name', chapter_dir.name)
        
        print(f"Adding {chapter_name}: {len(notes_data)} cards")
        
        for note_data in notes_data:
            fields = note_data.get('fields', [])
            tags = note_data.get('tags', [])
            
            if len(fields) >= 4:
                # Clean tags and add chapter tag to distinguish cards
                clean_tags = []
                for tag in tags:
                    clean_tag = str(tag).replace(' ', '_').replace('-', '_')
                    clean_tags.append(clean_tag)
                
                chapter_tag = f"Chapter_{chapter_dir.name[:2]}"
                tags_with_chapter = clean_tags + [chapter_tag]
                
                note = genanki.Note(
                    model=note_model,
                    fields=fields,
                    tags=tags_with_chapter
                )
                master_deck.add_note(note)
                total_cards += 1
    
    # Create master APKG file
    output_file = output_path / "NLP-Complete-Collection.apkg"
    genanki.Package(master_deck).write_to_file(str(output_file))
    
    print("=" * 60)
    print(f"🎯 Master APKG created!")
    print(f"📁 File: {output_file.absolute()}")
    print(f"📊 Total cards: {total_cards}")
    print(f"🏷️ Tags: Each card tagged with chapter number for organization")

def create_llm_master_apkg():
    """Create a single APKG file containing all LLM chapters"""
    base_path = Path('../../data/source/flashcards/LLM')
    output_path = Path('../../data/output/apkg_files')
    output_path.mkdir(exist_ok=True)
    
    if not base_path.exists():
        print(f"❌ Path {base_path} not found!")
        return
    
    note_model = create_note_model()
    
    # Create master deck
    master_deck = genanki.Deck(
        deck_id=random.randrange(1 << 30, 1 << 31),
        name="ML:LLM:Complete Collection"
    )
    
    print("🔄 Creating LLM master APKG with all chapters...")
    print("=" * 60)
    
    # Get all JSON files (1.json through 7.json)
    json_files = sorted([f for f in base_path.glob("*.json") if f.name.replace('.json', '').isdigit()])
    
    total_cards = 0
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        notes_data = data.get('notes', [])
        chapter_name = data.get('name', f"LLM Chapter {json_file.stem}")
        
        print(f"Adding {chapter_name}: {len(notes_data)} cards")
        
        for note_data in notes_data:
            fields = note_data.get('fields', [])
            tags = note_data.get('tags', [])
            
            if len(fields) >= 4:
                # Clean tags and add chapter tag to distinguish cards
                clean_tags = []
                for tag in tags:
                    clean_tag = str(tag).replace(' ', '_').replace('-', '_')
                    clean_tags.append(clean_tag)
                
                # Add chapter tag based on file number
                chapter_tag = f"LLM_Chapter_{json_file.stem}"
                tags_with_chapter = clean_tags + [chapter_tag]
                
                note = genanki.Note(
                    model=note_model,
                    fields=fields,
                    tags=tags_with_chapter
                )
                master_deck.add_note(note)
                total_cards += 1
    
    # Create master APKG file
    output_file = output_path / "LLM-Complete-Collection.apkg"
    genanki.Package(master_deck).write_to_file(str(output_file))
    
    print("=" * 60)
    print(f"🎯 LLM Master APKG created!")
    print(f"📁 File: {output_file.absolute()}")
    print(f"📊 Total cards: {total_cards}")
    print(f"🏷️ Tags: Each card tagged with chapter number for organization")

def create_directory_apkg(directory_name, collection_name):
    """Create a single APKG file from any directory containing JSON files"""
    base_path = Path(f'../../data/source/flashcards/{directory_name}')
    output_path = Path('../../data/output/apkg_files')
    output_path.mkdir(exist_ok=True)
    
    if not base_path.exists():
        print(f"❌ Path {base_path} not found!")
        return
    
    note_model = create_note_model()
    
    # Create master deck
    master_deck = genanki.Deck(
        deck_id=random.randrange(1 << 30, 1 << 31),
        name=f"ML:{collection_name}:Complete Collection"
    )
    
    print(f"🔄 Creating {collection_name} master APKG with all files...")
    print("=" * 60)
    
    # Get all JSON files in the directory
    json_files = sorted(list(base_path.glob("*.json")))
    
    if not json_files:
        print(f"❌ No JSON files found in {directory_name}")
        return
    
    total_cards = 0
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Error in {json_file.name}")
            print(f"   Line {e.lineno}, Column {e.colno}: {e.msg}")
            print(f"   Skipping this file...")
            continue
        except Exception as e:
            print(f"❌ Error reading {json_file.name}: {e}")
            continue
        
        notes_data = data.get('notes', [])
        chapter_name = data.get('name', f"{collection_name} {json_file.stem}")
        
        print(f"Adding {chapter_name}: {len(notes_data)} cards")
        
        for note_data in notes_data:
            fields = note_data.get('fields', [])
            tags = note_data.get('tags', [])
            
            if len(fields) >= 4:
                # Clean tags and add file tag to distinguish cards
                clean_tags = []
                for tag in tags:
                    clean_tag = str(tag).replace(' ', '_').replace('-', '_')
                    clean_tags.append(clean_tag)
                
                # Add file tag based on filename
                file_tag = f"{collection_name}_{json_file.stem}"
                tags_with_file = clean_tags + [file_tag]
                
                note = genanki.Note(
                    model=note_model,
                    fields=fields,
                    tags=tags_with_file
                )
                master_deck.add_note(note)
                total_cards += 1
    
    if total_cards == 0:
        print("❌ No valid cards found!")
        return
    
    # Create master APKG file
    output_file = output_path / f"{collection_name}-Complete-Collection.apkg"
    genanki.Package(master_deck).write_to_file(str(output_file))
    
    print("=" * 60)
    print(f"🎯 {collection_name} Master APKG created!")
    print(f"📁 File: {output_file.absolute()}")
    print(f"📊 Total cards: {total_cards}")
    print(f"🏷️ Tags: Each card tagged with filename for organization")

def list_available_directories():
    """List all available directories with JSON files"""
    flashcards_path = Path('../../data/source/flashcards')
    if not flashcards_path.exists():
        print("❌ Flashcards directory not found!")
        return []
    
    directories = []
    for item in flashcards_path.iterdir():
        if item.is_dir():
            json_files = list(item.glob("*.json"))
            if json_files:
                directories.append(item.name)
    
    return directories

if __name__ == "__main__":
    try:
        import genanki
        print("✅ genanki library found")
        
        print("Choose conversion option:")
        print("1. Individual APKG files (one per chapter) - Recommended")
        print("2. Master APKG file (all NLP chapters combined)")
        print("3. Master APKG file (all LLM chapters combined)")
        print("4. Auto-detect and convert any directory")
        
        choice = input("Enter choice (1, 2, 3, or 4): ").strip()
        
        if choice == "1":
            create_individual_apkg_files()
        elif choice == "2":
            create_master_apkg()
        elif choice == "3":
            create_llm_master_apkg()
        elif choice == "4":
            directories = list_available_directories()
            if not directories:
                print("❌ No directories with JSON files found!")
            else:
                print("\n📁 Available directories:")
                for i, dir_name in enumerate(directories, 1):
                    print(f"   {i}. {dir_name}")
                
                try:
                    dir_choice = int(input(f"\nSelect directory (1-{len(directories)}): ").strip())
                    if 1 <= dir_choice <= len(directories):
                        selected_dir = directories[dir_choice - 1]
                        collection_name = input(f"Collection name for '{selected_dir}' (default: {selected_dir}): ").strip()
                        if not collection_name:
                            collection_name = selected_dir
                        create_directory_apkg(selected_dir, collection_name)
                    else:
                        print("❌ Invalid selection!")
                except ValueError:
                    print("❌ Invalid input!")
        else:
            print("Invalid choice. Running individual files conversion...")
            create_individual_apkg_files()
            
    except ImportError:
        print("❌ genanki library not found")
        print("📥 Install with: pip install genanki")
        print("🔗 Or use alternative methods below...")
