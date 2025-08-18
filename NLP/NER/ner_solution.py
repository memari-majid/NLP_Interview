from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

try:
    import spacy
    # Try to load the model, download if not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("Install spacy: pip install spacy")
    print("Then run: python -m spacy download en_core_web_sm")
    raise


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using spaCy."""
    doc = nlp(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    
    # Remove duplicates while preserving order
    for label in entities:
        entities[label] = list(dict.fromkeys(entities[label]))
    
    return dict(entities)


def extract_custom_entities(text: str) -> Dict[str, List[str]]:
    """Extract custom entities like emails, phones, URLs using regex."""
    entities = defaultdict(list)
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['EMAIL'] = re.findall(email_pattern, text)
    
    # Phone pattern (US-style)
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
    phones = re.findall(phone_pattern, text)
    entities['PHONE'] = ['-'.join(groups) for groups in phones]
    
    # URL pattern
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    entities['URL'] = re.findall(url_pattern, text)
    
    # Money pattern (simple)
    money_pattern = r'\$[\d,]+\.?\d*|\b\d+\s*(?:dollars?|cents?|USD|EUR|GBP)\b'
    entities['MONEY_CUSTOM'] = re.findall(money_pattern, text, re.IGNORECASE)
    
    # Remove empty categories
    return {k: v for k, v in entities.items() if v}


def extract_entities_with_context(text: str, context_window: int = 30) -> Dict[str, List[Tuple[str, str]]]:
    """Extract entities with surrounding context."""
    doc = nlp(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        # Get context
        start = max(0, ent.start_char - context_window)
        end = min(len(text), ent.end_char + context_window)
        context = text[start:end].strip()
        
        # Mark entity in context
        entity_start = ent.start_char - start
        entity_end = ent.end_char - start
        context_marked = (
            context[:entity_start] + 
            f"[{context[entity_start:entity_end]}]" + 
            context[entity_end:]
        )
        
        entities[ent.label_].append((ent.text, context_marked))
    
    return dict(entities)


def extract_entity_relationships(text: str) -> List[Tuple[str, str, str]]:
    """Extract simple relationships between entities."""
    doc = nlp(text)
    relationships = []
    
    # Simple pattern: PERSON + verb + ORG
    for sent in doc.sents:
        entities_in_sent = [(ent.text, ent.label_, ent.start, ent.end) for ent in sent.ents]
        
        # Look for patterns
        for i, (ent1_text, ent1_label, _, _) in enumerate(entities_in_sent):
            for j, (ent2_text, ent2_label, _, _) in enumerate(entities_in_sent[i+1:], i+1):
                # Extract verb between entities
                if ent1_label == "PERSON" and ent2_label == "ORG":
                    # Find verb between entities
                    between_tokens = sent[entities_in_sent[i][3]:entities_in_sent[j][2]]
                    verbs = [token.text for token in between_tokens if token.pos_ == "VERB"]
                    if verbs:
                        relationships.append((ent1_text, verbs[0], ent2_text))
    
    return relationships


def resolve_entity_coreferences(text: str) -> Dict[str, List[str]]:
    """Simple coreference resolution for entities (e.g., 'Apple' -> 'Apple Inc.')"""
    doc = nlp(text)
    entities = extract_entities(text)
    
    # Simple heuristic: map shorter versions to longer versions
    entity_mapping = {}
    
    for label, ent_list in entities.items():
        # Sort by length
        sorted_ents = sorted(ent_list, key=len, reverse=True)
        for i, longer in enumerate(sorted_ents):
            for shorter in sorted_ents[i+1:]:
                if shorter.lower() in longer.lower() and shorter != longer:
                    entity_mapping[shorter] = longer
    
    # Apply mapping
    resolved = defaultdict(set)
    for label, ent_list in entities.items():
        for ent in ent_list:
            canonical = entity_mapping.get(ent, ent)
            resolved[label].add(canonical)
    
    return {k: list(v) for k, v in resolved.items()}


def extract_entities_with_confidence(text: str) -> Dict[str, List[Tuple[str, float]]]:
    """Extract entities with confidence scores (using spaCy's scores if available)."""
    doc = nlp(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        # SpaCy doesn't always provide confidence, so we'll simulate
        # In practice, you'd use a model that provides confidence scores
        confidence = 0.95 if ent.label_ in ["PERSON", "ORG", "GPE"] else 0.85
        
        # Lower confidence for single-word entities
        if len(ent.text.split()) == 1:
            confidence *= 0.9
            
        entities[ent.label_].append((ent.text, confidence))
    
    return dict(entities)


if __name__ == "__main__":
    # Example 1: Basic NER
    text1 = "Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976."
    print("Basic NER:")
    entities = extract_entities(text1)
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}: {entity_list}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Custom entities
    text2 = "Contact John at john.doe@email.com or call 555-123-4567. Visit https://example.com for $99.99 deals."
    print("Custom entity extraction:")
    custom_entities = extract_custom_entities(text2)
    for entity_type, entity_list in custom_entities.items():
        print(f"{entity_type}: {entity_list}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Entities with context
    text3 = "Microsoft CEO Satya Nadella announced new AI features at the Seattle conference."
    print("Entities with context:")
    entities_context = extract_entities_with_context(text3)
    for entity_type, entity_list in entities_context.items():
        for entity, context in entity_list:
            print(f"{entity_type}: {entity}")
            print(f"  Context: {context}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Entity relationships
    text4 = "Tim Cook leads Apple. Satya Nadella manages Microsoft. Jeff Bezos founded Amazon."
    print("Entity relationships:")
    relationships = extract_entity_relationships(text4)
    for subj, verb, obj in relationships:
        print(f"{subj} --{verb}--> {obj}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Coreference resolution
    text5 = "Apple announced new products. Apple Inc. is based in Cupertino. Tim Cook, CEO of Apple, presented the keynote."
    print("Coreference resolution:")
    resolved = resolve_entity_coreferences(text5)
    print("Original entities:", extract_entities(text5))
    print("Resolved entities:", resolved)
