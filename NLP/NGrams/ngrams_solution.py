from typing import List, Dict, Tuple
from collections import defaultdict, Counter

def build_bigram_model(texts: List[str]) -> Dict[str, Dict[str, float]]:
    """Build bigram language model with probabilities."""
    if not texts:
        return {}
    
    # Count bigrams
    bigram_counts = defaultdict(Counter)
    
    for text in texts:
        words = text.lower().split()
        
        # Add start token
        words = ['<START>'] + words + ['<END>']
        
        # Count bigrams
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            bigram_counts[w1][w2] += 1
    
    # Convert counts to probabilities
    model = {}
    for w1, w2_counts in bigram_counts.items():
        total = sum(w2_counts.values())
        model[w1] = {w2: count/total for w2, count in w2_counts.items()}
    
    return model

def generate_text(model: Dict, start_word: str = '<START>', length: int = 5) -> str:
    """Generate text using bigram model (deterministic - pick most probable)."""
    if start_word not in model:
        return ""
    
    words = []
    current_word = start_word
    
    for _ in range(length):
        if current_word not in model or not model[current_word]:
            break
        
        # Pick most probable next word
        next_word = max(model[current_word], key=model[current_word].get)
        
        if next_word == '<END>':
            break
        
        if next_word != '<START>':
            words.append(next_word)
        
        current_word = next_word
    
    return ' '.join(words)

def calculate_probability(model: Dict, text: str) -> float:
    """Calculate probability of text under the model."""
    words = ['<START>'] + text.lower().split() + ['<END>']
    
    prob = 1.0
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        
        if w1 in model and w2 in model[w1]:
            prob *= model[w1][w2]
        else:
            return 0.0  # Unseen bigram
    
    return prob

# Test
if __name__ == "__main__":
    # Training data
    training_texts = [
        "the cat sat on the mat",
        "the dog sat on the log", 
        "the cat and the dog"
    ]
    
    # Build model
    model = build_bigram_model(training_texts)
    
    print("Bigram model sample:")
    for word, next_words in list(model.items())[:3]:
        print(f"'{word}' -> {next_words}")
    
    # Generate text
    generated = generate_text(model, '<START>', length=6)
    print(f"\nGenerated text: '{generated}'")
    
    # Calculate probability
    test_text = "the cat sat"
    prob = calculate_probability(model, test_text)
    print(f"P('{test_text}') = {prob:.6f}")
    
    # Test unseen text
    unseen_text = "the elephant danced"
    prob_unseen = calculate_probability(model, unseen_text)
    print(f"P('{unseen_text}') = {prob_unseen}")
