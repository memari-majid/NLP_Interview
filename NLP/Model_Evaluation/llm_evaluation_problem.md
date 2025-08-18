# Problem: LLM Evaluation Metrics

**Time: 20 minutes**

Implement key evaluation metrics for large language models.

```python
def calculate_perplexity(model_probs: List[List[float]], 
                        target_tokens: List[int]) -> float:
    """
    Calculate perplexity - primary metric for language models.
    
    Perplexity = exp(-1/N * sum(log(p(token_i))))
    Lower perplexity = better model
    
    Args:
        model_probs: Probability distributions over vocabulary for each position
        target_tokens: Actual next tokens
        
    Returns:
        Perplexity value
    """
    pass

def compute_bleu_score(reference: str, candidate: str, n: int = 4) -> float:
    """
    Compute BLEU score for text generation evaluation.
    
    Measures n-gram overlap between reference and generated text.
    Used for translation, summarization evaluation.
    """
    pass
```

**Requirements:**
- Handle log probability calculations safely (avoid log(0))
- Implement n-gram precision for BLEU
- Add brevity penalty for BLEU
- Calculate confidence intervals for perplexity

**Follow-up:** What are limitations of perplexity? How do you evaluate instruction-following?
