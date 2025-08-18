# Problem: LSTM for Sentiment Analysis

Implement an LSTM-based sentiment classifier:
1. `build_lstm_model(vocab_size: int, embedding_dim: int = 100) -> Model`
2. `train_sentiment_model(texts: List[str], labels: List[int], epochs: int = 10) -> Model`
3. `predict_sentiment(model: Model, texts: List[str]) -> List[Tuple[int, float]]`

Example:
Input: "This movie was absolutely fantastic!"
Output: (1, 0.95)  # 1=positive, confidence=0.95

Requirements:
- Handle variable-length sequences
- Implement with PyTorch or TensorFlow
- Add attention mechanism for interpretability
- Compare with GRU and simple RNN

Follow-ups:
- Bidirectional LSTM
- Multi-layer architecture
- Gradient clipping for stability
