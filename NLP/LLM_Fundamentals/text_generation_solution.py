import numpy as np
import math
from typing import List, Dict, Tuple, Callable
import heapq

def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """Convert logits to probabilities with temperature scaling."""
    if temperature != 1.0:
        logits = [l / temperature for l in logits]
    
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    
    return [exp_l / sum_exp for exp_l in exp_logits]

def sample_token(probs: List[float], strategy: str = 'greedy', 
                top_k: int = 5, top_p: float = 0.9) -> int:
    """Sample next token using different strategies."""
    
    if strategy == 'greedy':
        return probs.index(max(probs))
    
    elif strategy == 'random':
        # Random sampling from full distribution
        return np.random.choice(len(probs), p=probs)
    
    elif strategy == 'top_k':
        # Sample from top k tokens only
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
        top_probs = [probs[i] for i in top_indices]
        
        # Renormalize
        sum_top = sum(top_probs)
        if sum_top > 0:
            top_probs = [p / sum_top for p in top_probs]
            selected_idx = np.random.choice(len(top_probs), p=top_probs)
            return top_indices[selected_idx]
        else:
            return 0
    
    elif strategy == 'top_p':
        # Nucleus sampling
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        
        cumsum = 0.0
        nucleus_indices = []
        
        for idx in sorted_indices:
            cumsum += probs[idx]
            nucleus_indices.append(idx)
            if cumsum >= top_p:
                break
        
        # Sample from nucleus
        nucleus_probs = [probs[i] for i in nucleus_indices]
        sum_nucleus = sum(nucleus_probs)
        
        if sum_nucleus > 0:
            nucleus_probs = [p / sum_nucleus for p in nucleus_probs]
            selected_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
            return nucleus_indices[selected_idx]
        else:
            return 0
    
    return 0

def generate_text(model_fn: Callable, prompt: str, vocab: Dict, 
                 max_length: int = 20, strategy: str = 'greedy',
                 temperature: float = 1.0, **kwargs) -> str:
    """Generate text using autoregressive language model."""
    
    # Convert prompt to tokens
    prompt_tokens = [vocab.get(word.lower(), vocab.get('<UNK>', 0)) 
                    for word in prompt.split()]
    
    if not prompt_tokens:
        prompt_tokens = [vocab.get('<START>', 0)]
    
    # Generate tokens iteratively
    current_tokens = prompt_tokens.copy()
    generated_words = prompt.split()
    
    # Reverse vocabulary for decoding
    id_to_word = {idx: word for word, idx in vocab.items()}
    
    for _ in range(max_length):
        # Get next token logits from model
        logits = model_fn(current_tokens)
        
        # Convert to probabilities
        probs = softmax(logits, temperature)
        
        # Sample next token
        next_token_id = sample_token(probs, strategy, 
                                   top_k=kwargs.get('top_k', 5),
                                   top_p=kwargs.get('top_p', 0.9))
        
        # Convert to word
        next_word = id_to_word.get(next_token_id, '<UNK>')
        
        # Stop if end token
        if next_word in ['<END>', '<EOS>', '</s>']:
            break
        
        # Add to sequence
        current_tokens.append(next_token_id)
        generated_words.append(next_word)
    
    return ' '.join(generated_words)

def beam_search(model_fn: Callable, prompt_tokens: List[int], 
               beam_width: int = 3, max_length: int = 10,
               vocab: Dict = None) -> List[Tuple[List[int], float]]:
    """Implement beam search for finding high-probability sequences."""
    
    # Initialize beam with prompt
    # Each beam entry: (sequence, log_probability)
    beams = [(prompt_tokens.copy(), 0.0)]
    
    for step in range(max_length):
        new_beams = []
        
        for sequence, log_prob in beams:
            # Get next token logits
            logits = model_fn(sequence)
            probs = softmax(logits)
            
            # Consider top beam_width tokens
            top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:beam_width]
            
            for token_id in top_indices:
                new_sequence = sequence + [token_id]
                new_log_prob = log_prob + math.log(probs[token_id] + 1e-10)
                
                new_beams.append((new_sequence, new_log_prob))
        
        # Keep only top beam_width beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    
    return beams

def mock_language_model(tokens: List[int]) -> List[float]:
    """Mock language model that returns random logits."""
    vocab_size = 50
    np.random.seed(sum(tokens) % 100)  # Deterministic based on input
    return np.random.randn(vocab_size).tolist()

# Test
if __name__ == "__main__":
    print("Text Generation with LLMs")
    print("=" * 30)
    
    # Create sample vocabulary
    vocab = {
        'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5,
        'dog': 6, 'ran': 7, 'fast': 8, 'very': 9, 'good': 10,
        '<START>': 0, '<END>': 11, '<UNK>': 12
    }
    
    prompt = "the cat"
    
    print(f"Prompt: '{prompt}'")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test different generation strategies
    strategies = ['greedy', 'random', 'top_k', 'top_p']
    
    for strategy in strategies:
        generated = generate_text(
            mock_language_model, prompt, vocab, 
            max_length=8, strategy=strategy,
            temperature=0.8, top_k=5, top_p=0.9
        )
        print(f"{strategy:8}: '{generated}'")
    
    print("\n" + "=" * 30)
    
    # Test beam search
    print("Beam Search Results:")
    
    prompt_tokens = [vocab[word] for word in prompt.split()]
    beams = beam_search(mock_language_model, prompt_tokens, beam_width=3, max_length=5, vocab=vocab)
    
    id_to_word = {idx: word for word, idx in vocab.items()}
    
    for i, (sequence, log_prob) in enumerate(beams):
        words = [id_to_word.get(token_id, '<UNK>') for token_id in sequence]
        text = ' '.join(words)
        probability = math.exp(log_prob)
        print(f"Beam {i+1}: '{text}' (prob: {probability:.4f})")
    
    print("\n" + "=" * 30)
    
    # Demonstrate temperature effects
    print("Temperature Effects:")
    
    # Mock logits with clear preference
    test_logits = [5.0, 2.0, 1.0, 0.5, 0.1]  # Strong preference for token 0
    
    temps = [0.1, 1.0, 2.0]
    
    for temp in temps:
        probs = softmax(test_logits, temperature=temp)
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        print(f"Temperature {temp}: max_prob={max(probs):.3f}, entropy={entropy:.3f}")
    
    print("\nLower temperature -> more focused (less random)")
    print("Higher temperature -> more diverse (more random)")
    
    print("\n" + "=" * 30)
    print("Generation Strategy Trade-offs:")
    print("• Greedy: Fast, deterministic, but can be repetitive")
    print("• Random: Diverse but may be incoherent")  
    print("• Top-k: Good balance of quality and diversity")
    print("• Top-p: Adaptive vocabulary based on confidence")
    print("• Beam search: Finds high-probability sequences")
    print("• Temperature: Controls randomness vs quality")
