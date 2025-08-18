# Problem: LLM Fine-tuning for Classification

**Time: 25 minutes**

Implement the key components for fine-tuning a pre-trained LLM for text classification.

```python
def add_classification_head(pretrained_model_dim: int, num_classes: int) -> Dict:
    """
    Add classification head to pretrained LLM.
    
    Args:
        pretrained_model_dim: Size of LLM output (e.g., 768 for BERT-base)
        num_classes: Number of target classes
        
    Returns:
        Classification weights and bias
    """
    pass

def compute_classification_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute cross-entropy loss for classification.
    
    Args:
        logits: Model outputs (batch_size, num_classes)
        labels: True labels (batch_size,) as class indices
        
    Returns:
        Cross-entropy loss value
    """
    pass

def freeze_layers(model_weights: Dict, freeze_ratio: float = 0.8) -> Dict:
    """
    Freeze bottom layers of pretrained model (keep top layers trainable).
    In practice, this means marking parameters as requires_grad=False.
    """
    pass
```

**Requirements:**
- Initialize classification head with proper scaling
- Implement stable cross-entropy loss with softmax
- Demonstrate layer freezing strategy
- Handle different learning rates for pretrained vs new layers

**Follow-up:** How would you implement LoRA for parameter-efficient fine-tuning?
