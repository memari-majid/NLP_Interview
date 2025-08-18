import math
from typing import List, Dict, Tuple

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

def tanh(x: float) -> float:
    return math.tanh(max(-500, min(500, x)))

def dot_product(vec1: List[float], vec2: List[float]) -> float:
    return sum(a * b for a, b in zip(vec1, vec2))

def lstm_cell(x_t: List[float], h_prev: List[float], c_prev: List[float],
              weights: Dict) -> Tuple[List[float], List[float]]:
    """Single LSTM cell forward pass."""
    hidden_size = len(h_prev)
    combined = x_t + h_prev
    
    # Gates: forget, input, output  
    f_t = [sigmoid(dot_product(combined, weights['Wf'][i]) + weights['bf'][i]) for i in range(hidden_size)]
    i_t = [sigmoid(dot_product(combined, weights['Wi'][i]) + weights['bi'][i]) for i in range(hidden_size)]
    o_t = [sigmoid(dot_product(combined, weights['Wo'][i]) + weights['bo'][i]) for i in range(hidden_size)]
    
    # Candidate cell state
    c_candidate = [tanh(dot_product(combined, weights['Wc'][i]) + weights['bc'][i]) for i in range(hidden_size)]
    
    # Update cell and hidden states
    c_t = [f_t[i] * c_prev[i] + i_t[i] * c_candidate[i] for i in range(hidden_size)]
    h_t = [o_t[i] * tanh(c_t[i]) for i in range(hidden_size)]
    
    return h_t, c_t

def lstm_sentiment(sequence: List[List[float]], weights: Dict) -> float:
    """Run LSTM over sequence and classify sentiment."""
    if not sequence:
        return 0.5
    
    hidden_size = len(weights['bf'])
    h_t = [0.0] * hidden_size
    c_t = [0.0] * hidden_size
    
    # Process sequence
    for x_t in sequence:
        h_t, c_t = lstm_cell(x_t, h_t, c_t, weights)
    
    # Final classification
    logit = dot_product(h_t, weights['W_output']) + weights['b_output']
    return sigmoid(logit)

# Test with sample data
if __name__ == "__main__":
    import random
    
    # Create sample weights
    hidden_size = 3
    input_size = 5
    combined_size = input_size + hidden_size
    
    weights = {
        'Wf': [[random.uniform(-0.1, 0.1) for _ in range(combined_size)] for _ in range(hidden_size)],
        'Wi': [[random.uniform(-0.1, 0.1) for _ in range(combined_size)] for _ in range(hidden_size)],
        'Wo': [[random.uniform(-0.1, 0.1) for _ in range(combined_size)] for _ in range(hidden_size)],
        'Wc': [[random.uniform(-0.1, 0.1) for _ in range(combined_size)] for _ in range(hidden_size)],
        'bf': [0.0] * hidden_size,
        'bi': [0.0] * hidden_size,
        'bo': [0.0] * hidden_size,
        'bc': [0.0] * hidden_size,
        'W_output': [random.uniform(-0.1, 0.1) for _ in range(hidden_size)],
        'b_output': 0.0
    }
    
    # Test sequence
    sequence = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # word 1
        [0.2, 0.3, 0.4, 0.5, 0.6],  # word 2
    ]
    
    prob = lstm_sentiment(sequence, weights)
    print(f"Sentiment probability: {prob:.3f}")
    print(f"Prediction: {'positive' if prob > 0.5 else 'negative'}")
