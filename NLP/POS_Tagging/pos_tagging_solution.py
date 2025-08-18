from typing import List, Tuple, Dict
from collections import defaultdict, Counter

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('universal_tagset', quiet=True)
except ImportError:
    print("Install nltk: pip install nltk")
    raise


def pos_tag_text(text: str) -> List[Tuple[str, str]]:
    """POS tag text using NLTK's default tagger (Penn Treebank tagset)."""
    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)


def extract_pos(tagged_text: List[Tuple[str, str]], pos_prefix: str) -> List[str]:
    """Extract words with specific POS tags.
    
    Args:
        tagged_text: List of (word, pos) tuples
        pos_prefix: POS tag prefix (e.g., 'NN' for nouns, 'VB' for verbs)
    """
    return [word for word, pos in tagged_text if pos.startswith(pos_prefix)]


def find_noun_phrases(tagged_text: List[Tuple[str, str]]) -> List[str]:
    """Extract simple noun phrases using POS patterns."""
    noun_phrases = []
    current_phrase = []
    
    # Simple pattern: (DT)? (JJ)* (NN)+
    for word, pos in tagged_text:
        if pos == 'DT':  # Determiner
            if current_phrase:
                noun_phrases.append(' '.join(current_phrase))
            current_phrase = [word]
        elif pos.startswith('JJ'):  # Adjective
            if current_phrase:
                current_phrase.append(word)
        elif pos.startswith('NN'):  # Noun
            current_phrase.append(word)
        else:
            if current_phrase and any(pos.startswith('NN') for _, pos in 
                                    [(w, p) for w, p in zip(current_phrase, 
                                     [t[1] for t in tagged_text])]):
                noun_phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))
    
    return noun_phrases


def analyze_word_ambiguity(word: str, corpus_name: str = 'brown') -> Dict[str, float]:
    """Analyze POS tag distribution for an ambiguous word."""
    from nltk.corpus import brown
    
    # Get all occurrences of the word with their tags
    word_lower = word.lower()
    pos_counts = Counter()
    
    for sent in brown.tagged_sents(tagset='universal')[:10000]:  # Sample for speed
        for token, pos in sent:
            if token.lower() == word_lower:
                pos_counts[pos] += 1
    
    total = sum(pos_counts.values())
    if total == 0:
        return {}
    
    return {pos: count/total for pos, count in pos_counts.items()}


def compare_taggers(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """Compare different POS taggers on the same text."""
    tokens = nltk.word_tokenize(text)
    
    results = {
        'default': nltk.pos_tag(tokens),
        'universal': nltk.pos_tag(tokens, tagset='universal')
    }
    
    # Try to use spaCy if available
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        results['spacy'] = [(token.text, token.pos_) for token in doc]
    except:
        pass
    
    return results


def pos_tag_with_confidence(text: str) -> List[Tuple[str, str, float]]:
    """POS tag with confidence scores (simplified version)."""
    # This is a simplified demo - real confidence would come from the model
    tagged = pos_tag_text(text)
    
    # Add mock confidence based on word frequency/ambiguity
    result = []
    for word, pos in tagged:
        # Common unambiguous words get high confidence
        if pos in ['DT', 'IN', 'CC', '.', ',']:
            confidence = 0.99
        # Potentially ambiguous words get lower confidence
        elif word.lower() in ['run', 'meeting', 'light', 'bank']:
            confidence = 0.75
        else:
            confidence = 0.90
        
        result.append((word, pos, confidence))
    
    return result


if __name__ == "__main__":
    # Example 1: Basic POS tagging
    text1 = "The quick brown fox jumps over the lazy dog."
    print("Basic POS tagging:")
    tagged = pos_tag_text(text1)
    for word, pos in tagged:
        print(f"{word:10} -> {pos}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Extract specific POS
    print("Extracted nouns:", extract_pos(tagged, 'NN'))
    print("Extracted verbs:", extract_pos(tagged, 'VB'))
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Find noun phrases
    text2 = "The beautiful red car and the old wooden house are for sale."
    tagged2 = pos_tag_text(text2)
    print(f"Text: {text2}")
    print("Noun phrases:", find_noun_phrases(tagged2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Ambiguous words
    ambiguous_sentence = "I saw her duck by the river bank."
    print(f"Ambiguous sentence: {ambiguous_sentence}")
    tagged_amb = pos_tag_text(ambiguous_sentence)
    for word, pos in tagged_amb:
        if word.lower() in ['duck', 'bank']:
            print(f"{word} -> {pos}")
            # Uncomment to see ambiguity analysis (slow)
            # print(f"  Ambiguity analysis: {analyze_word_ambiguity(word)}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Compare taggers
    test_text = "Meeting rooms are booked."
    print(f"Comparing taggers on: '{test_text}'")
    comparisons = compare_taggers(test_text)
    for tagger, results in comparisons.items():
        print(f"\n{tagger}: {results}")
