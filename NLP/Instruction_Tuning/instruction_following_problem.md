# Problem: Instruction Following Setup

**Time: 20 minutes**

Implement the data preparation and loss calculation for instruction fine-tuning.

```python
def format_instruction_data(instruction: str, response: str) -> str:
    """
    Format instruction-response pair for training.
    
    Standard format:
    "### Instruction:\n{instruction}\n### Response:\n{response}"
    
    Used for training LLMs to follow instructions like ChatGPT.
    """
    pass

def compute_instruction_loss(model_output: np.ndarray, 
                           target_tokens: List[int],
                           instruction_length: int) -> float:
    """
    Compute loss only on response tokens (not instruction tokens).
    
    Args:
        model_output: Logits for next token prediction (seq_len, vocab_size)
        target_tokens: True next tokens (seq_len,)
        instruction_length: Length of instruction (don't compute loss here)
        
    Returns:
        Loss averaged over response tokens only
    """
    pass
```

**Requirements:**
- Format data with clear instruction/response delimiters
- Mask instruction tokens during loss calculation
- Implement next-token prediction loss
- Handle variable-length instructions and responses

**Follow-up:** How would you implement RLHF (reinforcement learning from human feedback)?
