#!/usr/bin/env python3
"""
Convert NLP Theory Flashcards to Anki Deck

This script converts the theoretical NLP interview questions from JSON format
into an Anki-compatible deck with multiple card types for optimal learning.

Features:
- Multiple card types per concept (basic, reverse, cloze)
- Difficulty-based scheduling
- Category-based tags
- Related concepts linking
"""

import json
import csv
import os
import re
from typing import Dict, List, Any
import hashlib
from datetime import datetime


class TheoryToAnkiConverter:
    """Converts theoretical NLP flashcards to Anki format"""
    
    def __init__(self, input_file: str = 'data/nlp_theory_flashcards.json'):
        self.input_file = input_file
        self.cards = []
        self.stats = {
            'total_concepts': 0,
            'cards_generated': 0,
            'categories': set(),
            'difficulty_distribution': {'easy': 0, 'medium': 0, 'hard': 0}
        }
        
    def load_flashcards(self) -> Dict:
        """Load flashcards from JSON file"""
        with open(self.input_file, 'r') as f:
            return json.load(f)
    
    def generate_card_id(self, content: str) -> str:
        """Generate unique ID for card"""
        return hashlib.md5(content.encode()).hexdigest()[:10]
    
    def create_basic_card(self, flashcard: Dict) -> List[Dict]:
        """Create basic Q&A card"""
        cards = []
        
        # Basic card
        cards.append({
            'id': self.generate_card_id(flashcard['id'] + '_basic'),
            'type': 'Basic',
            'front': flashcard['question'],
            'back': flashcard['answer'],
            'tags': f"NLP::{flashcard['category']} difficulty::{flashcard['difficulty']}",
            'notes': ''
        })
        
        return cards
    
    def create_key_points_cards(self, flashcard: Dict) -> List[Dict]:
        """Create cards for key points if available"""
        cards = []
        
        if 'key_points' in flashcard and flashcard['key_points']:
            points_text = '\n'.join([f"• {point}" for point in flashcard['key_points']])
            cards.append({
                'id': self.generate_card_id(flashcard['id'] + '_keypoints'),
                'type': 'Basic',
                'front': f"What are the key points about {flashcard['question'].lower().replace('what is', '').replace('what are', '').strip()}?",
                'back': points_text,
                'tags': f"NLP::{flashcard['category']} difficulty::{flashcard['difficulty']} type::key_points",
                'notes': ''
            })
        
        return cards
    
    def create_comparison_cards(self, flashcard: Dict) -> List[Dict]:
        """Create comparison cards if the question involves comparison"""
        cards = []
        
        if 'vs' in flashcard['question'].lower() or 'compare' in flashcard['question'].lower():
            if 'comparison' in flashcard:
                comparison_text = self.format_dict_as_text(flashcard['comparison'])
                cards.append({
                    'id': self.generate_card_id(flashcard['id'] + '_comparison'),
                    'type': 'Basic',
                    'front': f"Compare: {flashcard['question']}",
                    'back': comparison_text,
                    'tags': f"NLP::{flashcard['category']} difficulty::{flashcard['difficulty']} type::comparison",
                    'notes': ''
                })
        
        return cards
    
    def create_formula_cards(self, flashcard: Dict) -> List[Dict]:
        """Create formula cards if available"""
        cards = []
        
        if 'formula' in flashcard:
            cards.append({
                'id': self.generate_card_id(flashcard['id'] + '_formula'),
                'type': 'Basic',
                'front': f"What is the formula for {flashcard['question'].lower().replace('what is', '').replace('explain', '').strip()}?",
                'back': flashcard['formula'],
                'tags': f"NLP::{flashcard['category']} difficulty::{flashcard['difficulty']} type::formula",
                'notes': ''
            })
        
        return cards
    
    def create_example_cards(self, flashcard: Dict) -> List[Dict]:
        """Create example cards if available"""
        cards = []
        
        if 'examples' in flashcard or 'example' in flashcard:
            examples = flashcard.get('examples', flashcard.get('example', {}))
            if isinstance(examples, dict):
                example_text = self.format_dict_as_text(examples)
            else:
                example_text = str(examples)
            
            cards.append({
                'id': self.generate_card_id(flashcard['id'] + '_examples'),
                'type': 'Basic',
                'front': f"Give examples of: {flashcard['question'].lower().replace('what is', '').replace('what are', '').strip()}",
                'back': example_text,
                'tags': f"NLP::{flashcard['category']} difficulty::{flashcard['difficulty']} type::examples",
                'notes': ''
            })
        
        return cards
    
    def create_use_case_cards(self, flashcard: Dict) -> List[Dict]:
        """Create use case cards"""
        cards = []
        
        if 'use_cases' in flashcard:
            if isinstance(flashcard['use_cases'], list):
                use_cases_text = '\n'.join([f"• {case}" for case in flashcard['use_cases']])
            else:
                use_cases_text = self.format_dict_as_text(flashcard['use_cases'])
            
            cards.append({
                'id': self.generate_card_id(flashcard['id'] + '_usecases'),
                'type': 'Basic',
                'front': f"When should you use {flashcard['question'].lower().replace('what is', '').replace('what are', '').strip()}?",
                'back': use_cases_text,
                'tags': f"NLP::{flashcard['category']} difficulty::{flashcard['difficulty']} type::use_cases",
                'notes': ''
            })
        
        return cards
    
    def create_cloze_cards(self, flashcard: Dict) -> List[Dict]:
        """Create cloze deletion cards for key concepts"""
        cards = []
        
        # Create cloze for the main answer if it's short enough
        if len(flashcard['answer']) < 200:
            # Find key terms to create cloze deletions
            key_terms = self.extract_key_terms(flashcard['answer'])
            if key_terms:
                cloze_text = flashcard['answer']
                for i, term in enumerate(key_terms[:3], 1):  # Max 3 clozes
                    cloze_text = cloze_text.replace(term, f"{{{{c{i}::{term}}}}}", 1)
                
                cards.append({
                    'id': self.generate_card_id(flashcard['id'] + '_cloze'),
                    'type': 'Cloze',
                    'front': cloze_text,
                    'back': '',  # Cloze cards don't need separate back
                    'tags': f"NLP::{flashcard['category']} difficulty::{flashcard['difficulty']} type::cloze",
                    'notes': flashcard['question']
                })
        
        return cards
    
    def create_follow_up_cards(self, flashcard: Dict) -> List[Dict]:
        """Create follow-up question cards"""
        cards = []
        
        if 'follow_up' in flashcard:
            cards.append({
                'id': self.generate_card_id(flashcard['id'] + '_followup'),
                'type': 'Basic',
                'front': flashcard['follow_up'],
                'back': f"Related to: {flashcard['question']}\n\nThink about: {flashcard.get('answer', '')[:100]}...",
                'tags': f"NLP::{flashcard['category']} difficulty::hard type::follow_up",
                'notes': 'This is a thinking question - consider the implications'
            })
        
        return cards
    
    def format_dict_as_text(self, d: Dict) -> str:
        """Format dictionary as readable text"""
        if not d:
            return ""
        
        lines = []
        for key, value in d.items():
            if isinstance(value, list):
                value_str = ', '.join(value)
            else:
                value_str = str(value)
            lines.append(f"• {key}: {value_str}")
        return '\n'.join(lines)
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key technical terms from text"""
        # Simple extraction of capitalized terms and technical words
        terms = []
        words = text.split()
        
        technical_keywords = [
            'embedding', 'attention', 'transformer', 'encoder', 'decoder',
            'token', 'vector', 'neural', 'loss', 'gradient', 'weight',
            'parameter', 'layer', 'model', 'training', 'inference'
        ]
        
        for word in words:
            clean_word = word.strip('.,!?();:')
            if clean_word.lower() in technical_keywords and clean_word not in terms:
                terms.append(word)  # Keep original with punctuation context
        
        return terms[:5]  # Return top 5 terms
    
    def process_flashcard(self, flashcard: Dict):
        """Process single flashcard into multiple Anki cards"""
        # Generate different card types
        cards = []
        
        # Basic Q&A card
        cards.extend(self.create_basic_card(flashcard))
        
        # Key points cards
        cards.extend(self.create_key_points_cards(flashcard))
        
        # Comparison cards
        cards.extend(self.create_comparison_cards(flashcard))
        
        # Formula cards
        cards.extend(self.create_formula_cards(flashcard))
        
        # Example cards
        cards.extend(self.create_example_cards(flashcard))
        
        # Use case cards
        cards.extend(self.create_use_case_cards(flashcard))
        
        # Cloze cards
        cards.extend(self.create_cloze_cards(flashcard))
        
        # Follow-up cards
        cards.extend(self.create_follow_up_cards(flashcard))
        
        # Update stats
        self.stats['total_concepts'] += 1
        self.stats['cards_generated'] += len(cards)
        self.stats['categories'].add(flashcard['category'])
        self.stats['difficulty_distribution'][flashcard['difficulty']] += 1
        
        self.cards.extend(cards)
    
    def convert(self):
        """Main conversion process"""
        print("Loading NLP theory flashcards...")
        data = self.load_flashcards()
        
        print(f"Processing {len(data['flashcards'])} concepts...")
        for flashcard in data['flashcards']:
            self.process_flashcard(flashcard)
        
        print(f"Generated {len(self.cards)} Anki cards from {self.stats['total_concepts']} concepts")
        
        # Save to CSV
        output_file = 'anki_deck_nlp_theory.csv'
        self.save_to_csv(output_file)
        
        # Print statistics
        self.print_statistics()
        
        return output_file
    
    def save_to_csv(self, output_file: str):
        """Save cards to CSV file compatible with Anki"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # Write header
            writer.writerow(['#separator:Tab'])
            writer.writerow(['#html:true'])
            writer.writerow(['#tags column:4'])
            writer.writerow(['#deck:NLP Theory Interview'])
            
            # Write cards
            for card in self.cards:
                if card['type'] == 'Cloze':
                    # For cloze cards, front contains the cloze text
                    writer.writerow([
                        card['front'],
                        card['tags'],
                        card['notes']
                    ])
                else:
                    # For basic cards
                    front_html = self.format_html(card['front'])
                    back_html = self.format_html(card['back'])
                    writer.writerow([
                        front_html,
                        back_html,
                        card['tags'],
                        card['notes']
                    ])
    
    def format_html(self, text: str) -> str:
        """Format text with HTML for better Anki rendering"""
        # Convert line breaks
        text = text.replace('\n', '<br>')
        
        # Bold key terms
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Format bullet points
        text = text.replace('• ', '&bull; ')
        
        # Format code/formulas
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        return text
    
    def print_statistics(self):
        """Print conversion statistics"""
        print("\n" + "="*50)
        print("CONVERSION STATISTICS")
        print("="*50)
        print(f"Total concepts processed: {self.stats['total_concepts']}")
        print(f"Total cards generated: {self.stats['cards_generated']}")
        print(f"Average cards per concept: {self.stats['cards_generated']/self.stats['total_concepts']:.1f}")
        print(f"\nCategories covered: {', '.join(sorted(self.stats['categories']))}")
        print(f"\nDifficulty distribution:")
        for level, count in self.stats['difficulty_distribution'].items():
            print(f"  - {level}: {count} concepts")
        print("\nCard types generated:")
        card_types = {}
        for card in self.cards:
            card_type = 'cloze' if card['type'] == 'Cloze' else card['tags'].split('type::')[-1].split()[0] if 'type::' in card['tags'] else 'basic'
            card_types[card_type] = card_types.get(card_type, 0) + 1
        for card_type, count in sorted(card_types.items()):
            print(f"  - {card_type}: {count} cards")


def main():
    """Main execution"""
    converter = TheoryToAnkiConverter()
    
    # Check if input file exists
    if not os.path.exists(converter.input_file):
        print(f"Error: Input file '{converter.input_file}' not found!")
        print("Please ensure nlp_theory_flashcards.json exists in the data/ directory")
        return
    
    # Convert to Anki
    output_file = converter.convert()
    
    print(f"\n✅ Anki deck successfully created: {output_file}")
    print("\nTo import into Anki:")
    print("1. Open Anki")
    print("2. File → Import")
    print(f"3. Select '{output_file}'")
    print("4. Ensure 'Allow HTML in fields' is checked")
    print("5. Click 'Import'")
    
    print("\nStudy tips:")
    print("- Start with 'easy' difficulty cards")
    print("- Review categories you're weakest in")
    print("- Use the cloze cards for active recall")
    print("- Follow-up questions are great for interviews!")


if __name__ == "__main__":
    main()