from typing import List, Optional

# Stemming
try:
    from nltk.stem import PorterStemmer
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    print("Install nltk: pip install nltk")
    raise

# Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
    """Convert Penn Treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default


def stem_words(words: List[str]) -> List[str]:
    """Apply Porter stemming to words."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word.lower()) for word in words]


def lemmatize_words(words: List[str], pos_tags: Optional[List[str]] = None) -> List[str]:
    """Lemmatize words with optional POS tags for better accuracy."""
    lemmatizer = WordNetLemmatizer()
    
    if pos_tags is None:
        # Simple lemmatization without POS
        return [lemmatizer.lemmatize(word.lower()) for word in words]
    
    # Lemmatize with POS tags
    lemmatized = []
    for word, pos in zip(words, pos_tags):
        wordnet_pos = get_wordnet_pos(pos)
        lemmatized.append(lemmatizer.lemmatize(word.lower(), pos=wordnet_pos))
    
    return lemmatized


def compare_methods(words: List[str]) -> dict:
    """Compare stemming vs lemmatization results."""
    import nltk
    
    # Get POS tags
    pos_tags = [pos for _, pos in nltk.pos_tag(words)]
    
    stemmed = stem_words(words)
    lemmatized_simple = lemmatize_words(words)
    lemmatized_pos = lemmatize_words(words, pos_tags)
    
    return {
        'original': words,
        'stemmed': stemmed,
        'lemmatized_simple': lemmatized_simple,
        'lemmatized_with_pos': lemmatized_pos,
        'pos_tags': pos_tags
    }


if __name__ == "__main__":
    # Example 1: Verb forms
    verb_forms = ["running", "ran", "runs", "runner"]
    print("Verb forms comparison:")
    results = compare_methods(verb_forms)
    for key, values in results.items():
        print(f"{key:20}: {values}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Mixed words
    mixed_words = ["better", "best", "meeting", "meetings", "universal", "university"]
    print("Mixed words comparison:")
    results = compare_methods(mixed_words)
    for key, values in results.items():
        print(f"{key:20}: {values}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Context matters
    print("Context example - 'meeting' as noun vs verb:")
    sentences = [
        ("We are meeting tomorrow", "VERB"),
        ("The meeting is tomorrow", "NOUN")
    ]
    
    for sent, expected_pos in sentences:
        words = sent.split()
        pos_tags = [pos for _, pos in nltk.pos_tag(words)]
        idx = words.index("meeting")
        lemmatized = lemmatize_words([words[idx]], [pos_tags[idx]])
        print(f"'{sent}' -> 'meeting' tagged as {pos_tags[idx]} -> lemmatized: {lemmatized[0]}")
