# Problem: Text Classification Pipeline

Build a complete text classification system:
1. `train_classifier(texts: List[str], labels: List[str]) -> ClassificationModel`
2. `predict(model: ClassificationModel, texts: List[str]) -> List[str]`
3. `evaluate_classifier(y_true: List[str], y_pred: List[str]) -> Dict[str, float]`

Example:
Training data:
- "This movie is fantastic!" -> "positive"  
- "Terrible experience, would not recommend" -> "negative"
- "It was okay, nothing special" -> "neutral"

Test: "Amazing film, loved it!" -> "positive"

Requirements:
- Implement with both traditional ML (TF-IDF + LogisticRegression) and deep learning
- Handle imbalanced classes
- Add confidence scores to predictions

Follow-ups:
- Multi-label classification
- Active learning for uncertain predictions
- Feature importance analysis
