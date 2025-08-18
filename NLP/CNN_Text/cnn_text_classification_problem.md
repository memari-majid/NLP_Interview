# Problem: CNN for Text Classification

Implement a 1D CNN for text classification:
1. `build_cnn_model(vocab_size: int, embedding_dim: int, num_classes: int) -> Model`
2. `train_cnn(model, texts: List[str], labels: List[int], epochs: int = 10) -> Model`
3. `extract_features(model, text: str) -> np.ndarray`
4. `visualize_filters(model) -> Dict[str, np.ndarray]`

Example:
Text: "This movie is absolutely fantastic!"
Prediction: "positive" (0.95 confidence)
Extracted features: 128-dim vector from conv layers

Requirements:
- Multiple filter sizes (3, 4, 5) for n-gram detection
- Max pooling and dropout layers
- Implement in PyTorch/TensorFlow
- Handle variable length sequences

Follow-ups:
- Character-level CNN
- Multi-channel CNN (static + dynamic embeddings)
- Attention over CNN features
