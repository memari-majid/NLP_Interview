# Problem: Simple LSTM for Sentiment

**Time: 25 minutes**

Implement a basic LSTM cell for sentiment classification.

```python
def lstm_cell(x_t: List[float], h_prev: List[float], c_prev: List[float],
              weights: Dict) -> Tuple[List[float], List[float]]:
    """
    Single LSTM cell forward pass.
    
    Args:
        x_t: Input at time t
        h_prev: Previous hidden state  
        c_prev: Previous cell state
        weights: {'Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc'}
        
    Returns:
        (h_t, c_t): New hidden and cell states
    """
    pass

def lstm_sentiment(sequence: List[List[float]], weights: Dict) -> float:
    """
    Run LSTM over sequence and classify sentiment.
    Return probability (0-1) of positive sentiment.
    """
    pass
```

**Requirements:**
- Implement forget, input, output gates using sigmoid
- Implement candidate cell state using tanh
- Process sequence step by step
- Final classification with sigmoid

**Simplifications:** Use lists instead of matrices, single layer

**Follow-up:** How would you handle variable-length sequences in practice?