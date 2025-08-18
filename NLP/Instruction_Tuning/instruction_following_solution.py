import numpy as np
from typing import List, Dict, Tuple

def format_instruction_data(instruction: str, response: str) -> str:
    """Format instruction-response pair for training."""
    return f"### Instruction:\n{instruction}\n### Response:\n{response}"

def tokenize_instruction_response(formatted_text: str, 
                                tokenizer_vocab: Dict[str, int]) -> Tuple[List[int], int]:
    """
    Tokenize formatted instruction-response and return instruction length.
    
    Returns:
        (token_ids, instruction_length)
    """
    # Simple tokenization for demo
    tokens = formatted_text.lower().split()
    token_ids = [tokenizer_vocab.get(token, 0) for token in tokens]  # 0 = UNK
    
    # Find where instruction ends (look for "### Response:")
    instruction_length = 0
    response_marker = "### response:"
    
    # Reconstruct text to find marker
    text_lower = formatted_text.lower()
    if response_marker in text_lower:
        before_response = text_lower[:text_lower.index(response_marker)]
        instruction_tokens = before_response.split()
        instruction_length = len(instruction_tokens) + 2  # +2 for "### response:"
    
    return token_ids, instruction_length

def compute_instruction_loss(model_output: np.ndarray, 
                           target_tokens: List[int],
                           instruction_length: int) -> float:
    """Compute loss only on response tokens."""
    seq_len, vocab_size = model_output.shape
    
    if instruction_length >= seq_len:
        return 0.0  # No response tokens to train on
    
    # Only compute loss on response portion
    response_logits = model_output[instruction_length:]
    response_targets = target_tokens[instruction_length:]
    
    if len(response_targets) == 0:
        return 0.0
    
    # Cross-entropy loss on response tokens only
    loss = 0.0
    num_response_tokens = len(response_targets)
    
    for i, target_token in enumerate(response_targets):
        if i < len(response_logits):
            # Softmax
            logits = response_logits[i]
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            
            # Cross-entropy for this token
            loss += -np.log(probs[target_token] + 1e-10)
    
    return loss / num_response_tokens if num_response_tokens > 0 else 0.0

def create_instruction_dataset(examples: List[Tuple[str, str]]) -> List[Dict]:
    """Create instruction tuning dataset."""
    dataset = []
    
    for instruction, response in examples:
        formatted = format_instruction_data(instruction, response)
        
        # Simple tokenizer for demo
        vocab = {word: i for i, word in enumerate(set(formatted.lower().split()))}
        vocab['<UNK>'] = 0
        
        token_ids, inst_len = tokenize_instruction_response(formatted, vocab)
        
        dataset.append({
            'formatted_text': formatted,
            'token_ids': token_ids,
            'instruction_length': inst_len,
            'vocab': vocab
        })
    
    return dataset

def evaluate_instruction_following(model_responses: List[str], 
                                 instructions: List[str]) -> Dict[str, float]:
    """Simple evaluation metrics for instruction following."""
    
    # Instruction following metrics (simplified)
    scores = {
        'avg_response_length': np.mean([len(response.split()) for response in model_responses]),
        'response_rate': sum(1 for response in model_responses if len(response.strip()) > 0) / len(model_responses),
        'keyword_compliance': 0.0
    }
    
    # Check if response contains key instruction words
    keyword_matches = 0
    total_keywords = 0
    
    for instruction, response in zip(instructions, model_responses):
        # Extract important words from instruction (simplified)
        inst_words = set(instruction.lower().split())
        resp_words = set(response.lower().split())
        
        # Remove common words
        important_words = inst_words - {'the', 'a', 'an', 'is', 'are', 'and', 'or', 'but'}
        
        if important_words:
            overlap = len(important_words & resp_words) / len(important_words)
            keyword_matches += overlap
            total_keywords += 1
    
    if total_keywords > 0:
        scores['keyword_compliance'] = keyword_matches / total_keywords
    
    return scores

def sampling_strategies(logits: np.ndarray, temperature: float = 1.0, 
                       top_k: int = None, top_p: float = None) -> int:
    """Implement different sampling strategies for text generation."""
    
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Convert to probabilities
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    
    # Top-k sampling
    if top_k is not None:
        top_k_indices = np.argsort(probs)[-top_k:]
        masked_probs = np.zeros_like(probs)
        masked_probs[top_k_indices] = probs[top_k_indices]
        probs = masked_probs / np.sum(masked_probs)
    
    # Top-p (nucleus) sampling  
    if top_p is not None:
        sorted_indices = np.argsort(probs)[::-1]
        cumsum_probs = np.cumsum(probs[sorted_indices])
        
        # Find cutoff where cumulative probability exceeds top_p
        cutoff_idx = np.argmax(cumsum_probs >= top_p) + 1
        
        # Keep only top-p tokens
        top_p_indices = sorted_indices[:cutoff_idx]
        masked_probs = np.zeros_like(probs)
        masked_probs[top_p_indices] = probs[top_p_indices]
        probs = masked_probs / np.sum(masked_probs)
    
    # Sample from distribution
    return np.random.choice(len(probs), p=probs)

# Test
if __name__ == "__main__":
    print("Instruction Tuning Implementation")
    print("=" * 40)
    
    # Example instruction-response pairs
    examples = [
        ("Summarize this text: 'The quick brown fox jumps over the lazy dog.'", 
         "A fox jumps over a dog."),
        ("What is the capital of France?", 
         "The capital of France is Paris."),
        ("Translate 'hello' to Spanish", 
         "Hello in Spanish is 'hola'.")
    ]
    
    # Create dataset
    dataset = create_instruction_dataset(examples)
    
    print("Sample formatted data:")
    print(dataset[0]['formatted_text'])
    print(f"\nInstruction length: {dataset[0]['instruction_length']} tokens")
    print(f"Total tokens: {len(dataset[0]['token_ids'])}")
    
    # Simulate training step
    print("\n" + "=" * 40)
    print("Training Step Simulation")
    
    # Mock model output and targets
    seq_len, vocab_size = 15, 100
    mock_logits = np.random.randn(seq_len, vocab_size)
    mock_targets = np.random.randint(0, vocab_size, seq_len)
    instruction_len = 8
    
    # Compute loss only on response tokens
    loss = compute_instruction_loss(mock_logits, mock_targets, instruction_len)
    print(f"Instruction tuning loss: {loss:.4f}")
    print(f"Loss computed on {seq_len - instruction_len} response tokens only")
    
    # Demonstrate sampling strategies
    print("\n" + "=" * 40)
    print("Text Generation Sampling")
    
    # Sample logits for next token prediction
    sample_logits = np.array([2.0, 1.0, 0.5, 0.2, 0.1])  # 5 possible tokens
    
    print("Sample next-token logits:", sample_logits)
    
    # Different sampling methods
    strategies = [
        ("greedy", {"temperature": 0.01}),
        ("random", {"temperature": 1.0}),
        ("top_k", {"temperature": 1.0, "top_k": 3}),
        ("top_p", {"temperature": 1.0, "top_p": 0.8})
    ]
    
    for name, params in strategies:
        token = sampling_strategies(sample_logits, **params)
        print(f"{name:8} sampling -> token {token}")
    
    print("\n" + "=" * 40)
    print("Key Instruction Tuning Concepts:")
    print("• Only compute loss on response tokens")
    print("• Use special formatting to separate instruction/response")
    print("• Different sampling strategies affect generation quality")
    print("• Lower learning rates prevent catastrophic forgetting")
    print("• Evaluation requires human judgment or automated metrics")
