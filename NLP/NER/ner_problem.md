# Problem: Named Entity Recognition with Custom Entities

Implement `extract_entities(text: str) -> Dict[str, List[str]]` that:
1. Extracts standard entities (PERSON, ORG, GPE, DATE, MONEY)
2. Returns entities grouped by type
3. Handles overlapping entities

Example:
Input: "Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976."
Output: {
    "ORG": ["Apple Inc."],
    "PERSON": ["Steve Jobs"],
    "GPE": ["Cupertino"],
    "DATE": ["April 1, 1976"]
}

Requirements:
- Use spaCy or NLTK for NER
- Handle multi-word entities
- Implement custom entity detection for email/phone numbers

Follow-ups:
- Add confidence scores for entities
- Implement entity linking/disambiguation
- Extract relationships between entities
