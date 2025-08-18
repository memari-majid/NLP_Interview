# Problem: Text Generation with LLMs

**Time: 25 minutes**

Implement text generation from a trained language model with different decoding strategies.

```python
def generate_text(model_fn: callable, prompt: str, vocab: Dict, 
                 max_length: int = 20, strategy: str = 'greedy') -> str:
    """
    Generate text from language model using different strategies.
    
    Args:
        model_fn: Function that returns next-token logits given current sequence
        prompt: Starting text
        vocab: Token to ID mapping
        max_length: Maximum tokens to generate  
        strategy: 'greedy', 'random', 'top_k', or 'top_p'
        
    Returns:
        Generated text
    """
    pass

def beam_search(model_fn: callable, prompt_tokens: List[int], 
               beam_width: int = 3, max_length: int = 10) -> List[str]:
    """
    Implement beam search for finding high-probability sequences.
    
    Returns:
        List of candidate sequences ranked by score
    """
    pass
```

**Requirements:**
- Convert prompt to tokens, generate tokens iteratively
- Implement greedy, random, top-k, and top-p sampling
- Track probabilities for beam search scoring
- Handle end-of-sequence tokens properly

**Follow-up:** How do you balance quality vs diversity in generation?
