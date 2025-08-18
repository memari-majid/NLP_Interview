import numpy as np
from typing import Dict, List, Tuple

def add_classification_head(pretrained_model_dim: int, num_classes: int) -> Dict:
    """Add classification head to pretrained LLM."""
    
    # Xavier/Glorot initialization for stable training
    std = np.sqrt(2.0 / (pretrained_model_dim + num_classes))
    
    return {
        'W_cls': np.random.randn(pretrained_model_dim, num_classes) * std,
        'b_cls': np.zeros(num_classes)
    }

def compute_classification_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute cross-entropy loss with numerical stability."""
    batch_size, num_classes = logits.shape
    
    # Numerical stability: subtract max from logits
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    
    # Softmax probabilities
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Cross-entropy loss
    loss = 0.0
    for i in range(batch_size):
        true_class = labels[i]
        loss += -np.log(probs[i, true_class] + 1e-10)  # Add small epsilon
    
    return loss / batch_size

def freeze_layers(model_weights: Dict, freeze_ratio: float = 0.8) -> Dict:
    """Mark layers as frozen (simulate requires_grad=False)."""
    frozen_info = {}
    
    # Sort layers by name to freeze bottom layers
    layer_names = sorted([name for name in model_weights.keys() if 'layer_' in name])
    
    num_layers = len(layer_names)
    num_frozen = int(num_layers * freeze_ratio)
    
    for i, layer_name in enumerate(layer_names):
        frozen_info[layer_name] = i < num_frozen  # True if frozen
    
    # Never freeze classification head
    for name in model_weights.keys():
        if 'cls' in name:
            frozen_info[name] = False
    
    return frozen_info

def fine_tuning_step(x: np.ndarray, labels: np.ndarray, 
                    pretrained_weights: Dict, cls_head: Dict,
                    learning_rates: Dict) -> Tuple[float, Dict]:
    """Single fine-tuning step (forward + backward)."""
    
    # Forward pass through pretrained model (simplified)
    # In practice, this would be the full LLM forward pass
    pretrained_output = x @ pretrained_weights['final_layer'] + pretrained_weights['final_bias']
    
    # Classification head forward pass
    logits = pretrained_output @ cls_head['W_cls'] + cls_head['b_cls']
    
    # Compute loss
    loss = compute_classification_loss(logits, labels)
    
    # Backward pass (simplified gradients)
    batch_size = x.shape[0]
    
    # Softmax for gradient calculation
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Gradient w.r.t logits
    grad_logits = probs.copy()
    for i in range(batch_size):
        grad_logits[i, labels[i]] -= 1
    grad_logits /= batch_size
    
    # Gradients for classification head
    grad_W_cls = pretrained_output.T @ grad_logits
    grad_b_cls = np.sum(grad_logits, axis=0)
    
    # Update classification head with higher learning rate
    cls_head['W_cls'] -= learning_rates['cls_head'] * grad_W_cls
    cls_head['b_cls'] -= learning_rates['cls_head'] * grad_b_cls
    
    # Update pretrained layers with lower learning rate (if not frozen)
    grad_pretrained = grad_logits @ cls_head['W_cls'].T
    
    if not pretrained_weights.get('frozen', True):
        pretrained_weights['final_layer'] -= learning_rates['pretrained'] * (x.T @ grad_pretrained)
        pretrained_weights['final_bias'] -= learning_rates['pretrained'] * np.sum(grad_pretrained, axis=0)
    
    return loss, {'grad_W_cls': grad_W_cls, 'grad_b_cls': grad_b_cls}

def compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy."""
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels)

def lora_approximation(weight_matrix: np.ndarray, rank: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate LoRA (Low-Rank Adaptation) decomposition.
    
    Instead of updating full weight matrix W, update W + A*B where:
    - A is (d, rank) and B is (rank, d) 
    - Only A and B are trainable (much fewer parameters)
    """
    d_in, d_out = weight_matrix.shape
    
    # Initialize LoRA matrices
    A = np.random.randn(d_in, rank) * 0.01
    B = np.zeros((rank, d_out))  # B initialized to zero
    
    return A, B

# Test the fine-tuning process
def test_fine_tuning():
    """Demonstrate fine-tuning workflow."""
    
    # Sample data
    batch_size, seq_len, d_model = 8, 10, 64
    num_classes = 3
    
    # Simulated pretrained model output
    x = np.random.randn(batch_size, d_model)  # [CLS] token representations
    labels = np.random.randint(0, num_classes, batch_size)
    
    print("Fine-tuning Simulation")
    print("=" * 30)
    print(f"Batch size: {batch_size}")
    print(f"Model dimension: {d_model}")
    print(f"Number of classes: {num_classes}")
    
    # Add classification head
    cls_head = add_classification_head(d_model, num_classes)
    print(f"\nClassification head shape: {cls_head['W_cls'].shape}")
    
    # Create pretrained weights
    pretrained_weights = {
        'final_layer': np.random.randn(d_model, d_model) * 0.02,
        'final_bias': np.zeros(d_model),
        'frozen': False  # Will be set by freeze_layers
    }
    
    # Demonstrate freezing
    freeze_info = freeze_layers({'layer_0': None, 'layer_1': None, 'layer_2': None}, freeze_ratio=0.67)
    print(f"\nLayer freezing (freeze_ratio=0.67): {freeze_info}")
    
    # Different learning rates
    learning_rates = {
        'pretrained': 1e-5,  # Lower LR for pretrained layers
        'cls_head': 1e-3     # Higher LR for new classification head
    }
    
    print(f"\nLearning rates: {learning_rates}")
    
    # Training loop simulation
    print(f"\nTraining simulation:")
    for epoch in range(3):
        loss, grads = fine_tuning_step(x, labels, pretrained_weights, cls_head, learning_rates)
        
        # Calculate accuracy
        logits = x @ cls_head['W_cls'] + cls_head['b_cls']
        accuracy = compute_accuracy(logits, labels)
        
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.3f}")
    
    # Demonstrate LoRA
    print(f"\n" + "=" * 30)
    print("LoRA Parameter-Efficient Fine-tuning")
    
    # Original weight matrix
    W_original = np.random.randn(512, 512)
    A, B = lora_approximation(W_original, rank=8)
    
    original_params = W_original.size
    lora_params = A.size + B.size
    reduction = (1 - lora_params / original_params) * 100
    
    print(f"Original parameters: {original_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Parameter reduction: {reduction:.1f}%")
    
    # Show how LoRA update works
    lora_update = A @ B
    updated_weight = W_original + lora_update
    print(f"LoRA update shape: {lora_update.shape}")
    print("✓ LoRA allows efficient adaptation with few parameters")

if __name__ == "__main__":
    test_fine_tuning()
    
    print(f"\n" + "=" * 45)
    print("Fine-tuning Best Practices:")
    print("• Use lower learning rates for pretrained layers")
    print("• Freeze early layers, fine-tune later layers")  
    print("• Initialize new layers carefully (Xavier/Glorot)")
    print("• Monitor both loss and task-specific metrics")
    print("• Consider parameter-efficient methods (LoRA, adapters)")
    print("• Use gradient clipping to prevent instability")
