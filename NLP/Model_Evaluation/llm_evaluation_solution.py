import math
from typing import List, Dict, Set
from collections import Counter

def calculate_perplexity(model_probs: List[List[float]], 
                        target_tokens: List[int]) -> float:
    """Calculate perplexity from model probabilities."""
    if not model_probs or not target_tokens:
        return float('inf')
    
    if len(model_probs) != len(target_tokens):
        return float('inf')
    
    log_likelihood = 0.0
    num_tokens = 0
    
    for probs, target_token in zip(model_probs, target_tokens):
        if target_token < len(probs):
            # Get probability of target token
            prob = probs[target_token]
            
            # Add log probability (with small epsilon to avoid log(0))
            log_likelihood += math.log(max(prob, 1e-10))
            num_tokens += 1
    
    if num_tokens == 0:
        return float('inf')
    
    # Perplexity = exp(-avg_log_likelihood)
    avg_log_likelihood = log_likelihood / num_tokens
    perplexity = math.exp(-avg_log_likelihood)
    
    return perplexity

def get_ngrams(text: str, n: int) -> List[str]:
    """Extract n-grams from text."""
    words = text.lower().split()
    ngrams = []
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)
    
    return ngrams

def compute_bleu_score(reference: str, candidate: str, n: int = 4) -> float:
    """Compute BLEU score for text generation evaluation."""
    
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if not cand_words:
        return 0.0
    
    # Brevity penalty
    ref_len = len(ref_words)
    cand_len = len(cand_words)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Calculate n-gram precisions
    precisions = []
    
    for i in range(1, n + 1):
        ref_ngrams = Counter(get_ngrams(reference, i))
        cand_ngrams = Counter(get_ngrams(candidate, i))
        
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        
        # Count matches (with clipping for multiple references)
        matches = 0
        for ngram, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        precision = matches / sum(cand_ngrams.values())
        precisions.append(precision)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geometric_mean = 0.0
    
    return bp * geometric_mean

def calculate_cross_entropy_loss(logits: List[List[float]], 
                               targets: List[int]) -> float:
    """Calculate cross-entropy loss from logits."""
    if not logits or not targets or len(logits) != len(targets):
        return float('inf')
    
    total_loss = 0.0
    
    for logit_vec, target in zip(logits, targets):
        if target < len(logit_vec):
            # Softmax with numerical stability
            max_logit = max(logit_vec)
            exp_logits = [math.exp(logit - max_logit) for logit in logit_vec]
            sum_exp = sum(exp_logits)
            
            # Probability of target token
            prob = exp_logits[target] / sum_exp
            
            # Cross-entropy
            total_loss += -math.log(max(prob, 1e-10))
    
    return total_loss / len(targets)

def evaluate_generation_quality(references: List[str], 
                               candidates: List[str]) -> Dict[str, float]:
    """Comprehensive evaluation of generated text quality."""
    
    if len(references) != len(candidates):
        raise ValueError("Number of references and candidates must match")
    
    # Calculate average BLEU scores
    bleu_scores = []
    for ref, cand in zip(references, candidates):
        bleu = compute_bleu_score(ref, cand)
        bleu_scores.append(bleu)
    
    # Calculate other metrics
    length_ratios = []
    for ref, cand in zip(references, candidates):
        ref_len = len(ref.split())
        cand_len = len(cand.split())
        ratio = cand_len / ref_len if ref_len > 0 else 0
        length_ratios.append(ratio)
    
    return {
        'avg_bleu': sum(bleu_scores) / len(bleu_scores),
        'avg_length_ratio': sum(length_ratios) / len(length_ratios),
        'min_bleu': min(bleu_scores),
        'max_bleu': max(bleu_scores)
    }

# Test functions
if __name__ == "__main__":
    print("LLM Evaluation Metrics")
    print("=" * 30)
    
    # Test perplexity calculation
    print("1. Perplexity Calculation")
    
    # Mock model probabilities (3 positions, 5 tokens in vocab)
    model_probs = [
        [0.1, 0.6, 0.2, 0.05, 0.05],  # Position 0: high prob for token 1
        [0.3, 0.1, 0.4, 0.1, 0.1],    # Position 1: high prob for token 2  
        [0.2, 0.2, 0.2, 0.3, 0.1]     # Position 2: high prob for token 3
    ]
    
    target_tokens = [1, 2, 3]  # Actual next tokens
    
    perplexity = calculate_perplexity(model_probs, target_tokens)
    print(f"Perplexity: {perplexity:.2f}")
    print("(Lower is better - perfect prediction would be 1.0)")
    
    # Test with poor predictions
    poor_probs = [
        [0.2, 0.2, 0.2, 0.2, 0.2],  # Uniform (uncertain)
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]
    
    poor_perplexity = calculate_perplexity(poor_probs, target_tokens)
    print(f"Poor model perplexity: {poor_perplexity:.2f}")
    
    print("\n" + "=" * 30)
    
    # Test BLEU score
    print("2. BLEU Score Calculation")
    
    reference = "The quick brown fox jumps over the lazy dog"
    candidates = [
        "The quick brown fox jumps over the lazy dog",  # Perfect match
        "A quick brown fox jumps over a lazy dog",      # Good match
        "The fox jumps over the dog",                   # Shorter but relevant
        "Hello world this is different"                 # Poor match
    ]
    
    for i, candidate in enumerate(candidates):
        bleu = compute_bleu_score(reference, candidate)
        print(f"Candidate {i+1}: BLEU = {bleu:.3f}")
        print(f"  '{candidate}'")
    
    print("\n" + "=" * 30)
    
    # Test generation evaluation
    print("3. Generation Quality Evaluation")
    
    refs = [
        "The weather is nice today",
        "I love programming in Python"
    ]
    
    cands = [
        "Today the weather is quite nice",
        "Python programming is something I really enjoy"
    ]
    
    quality_metrics = evaluate_generation_quality(refs, cands)
    print("Quality metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n" + "=" * 30)
    
    # Test sampling strategies
    print("4. Sampling Strategies")
    
    # Mock next-token logits
    logits = [3.0, 2.0, 1.0, 0.5, 0.1]  # 5 possible next tokens
    print(f"Input logits: {logits}")
    
    # Test different sampling
    for _ in range(3):
        greedy = sampling_strategies(logits, temperature=0.01)  # Nearly deterministic
        random_sample = sampling_strategies(logits, temperature=1.0)
        top_k_sample = sampling_strategies(logits, temperature=1.0, top_k=3)
        top_p_sample = sampling_strategies(logits, temperature=1.0, top_p=0.8)
        
        print(f"Greedy: {greedy}, Random: {random_sample}, Top-k: {top_k_sample}, Top-p: {top_p_sample}")
    
    print("\n" + "=" * 30)
    print("Evaluation Summary:")
    print("• Perplexity: Measures how well model predicts next tokens")
    print("• BLEU: Measures n-gram overlap with reference text")
    print("• Sampling strategies affect generation diversity")
    print("• Human evaluation often needed for instruction following")
