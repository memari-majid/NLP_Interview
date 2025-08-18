import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import math

def extract_features(texts: List[str], method: str = 'tfidf') -> Tuple[List[List[float]], List[str]]:
    """
    Extract features from texts for classification.
    
    This is the MOST IMPORTANT step in text classification.
    Feature quality determines model performance more than algorithm choice.
    """
    
    # STEP 1: Build vocabulary from all texts
    # This creates our feature space - each unique word becomes a dimension
    vocab = set()
    for text in texts:
        words = text.lower().split()  # Simple tokenization
        vocab.update(words)
    vocab = sorted(list(vocab))  # Sort for consistency
    
    if method == 'tfidf':
        # STEP 2: Calculate document frequencies for IDF computation
        # DF = number of documents containing each term
        doc_freq = {}
        for word in vocab:
            doc_freq[word] = sum(1 for text in texts if word in text.lower())
        
        # STEP 3: Convert each text to TF-IDF vector
        feature_matrix = []
        for text in texts:
            words = text.lower().split()
            word_counts = Counter(words)
            doc_length = len(words)
            
            # Calculate TF-IDF for each vocabulary word
            tfidf_vector = []
            for word in vocab:
                # TF: How often word appears in this document (normalized)
                tf = word_counts[word] / doc_length if doc_length > 0 else 0
                
                # IDF: How rare the word is across all documents
                idf = math.log(len(texts) / doc_freq[word])
                
                # TF-IDF: Combines local importance (TF) with global rarity (IDF)
                tfidf_score = tf * idf
                tfidf_vector.append(tfidf_score)
            
            feature_matrix.append(tfidf_vector)
        
        return feature_matrix, vocab
    
    else:  # Simple bag-of-words
        feature_matrix = []
        for text in texts:
            word_counts = Counter(text.lower().split())
            bow_vector = [word_counts[word] for word in vocab]
            feature_matrix.append(bow_vector)
        
        return feature_matrix, vocab

def train_logistic_regression(X: List[List[float]], y: List[int]) -> Dict:
    """
    Train logistic regression classifier from scratch.
    
    Logistic regression is linear classifier with sigmoid activation.
    Good baseline for text classification - simple but effective.
    """
    # STEP 1: Convert to numpy arrays for easier math
    X_np = np.array(X)
    y_np = np.array(y)
    
    # STEP 2: Add bias term (intercept)
    # This allows the decision boundary to not pass through origin
    X_with_bias = np.column_stack([np.ones(len(X)), X_np])
    
    # STEP 3: Initialize weights randomly (small values)
    # Small initialization prevents sigmoid saturation early in training
    n_features = X_with_bias.shape[1]
    weights = np.random.randn(n_features) * 0.01
    
    # STEP 4: Training loop using gradient descent
    learning_rate = 0.01  # Step size for weight updates
    epochs = 100          # Number of training iterations
    
    for epoch in range(epochs):
        # FORWARD PASS: Compute predictions
        # Linear combination followed by sigmoid activation
        logits = X_with_bias @ weights           # Linear part: Xw + b
        predictions = 1 / (1 + np.exp(-logits)) # Sigmoid: maps to [0,1]
        
        # BACKWARD PASS: Compute gradients
        # Gradient of cross-entropy loss w.r.t. weights
        errors = predictions - y_np              # Prediction errors
        gradients = (X_with_bias.T @ errors) / len(X)  # Average gradient
        
        # UPDATE WEIGHTS: Move in opposite direction of gradient
        weights -= learning_rate * gradients
    
    return {'weights': weights, 'type': 'logistic'}

def predict_logistic(X: List[List[float]], model: Dict) -> List[int]:
    """Make predictions with trained logistic regression model."""
    X_np = np.array(X)
    
    # Add bias term (same as training)
    X_with_bias = np.column_stack([np.ones(len(X)), X_np])
    
    # Calculate probabilities
    logits = X_with_bias @ model['weights']
    probabilities = 1 / (1 + np.exp(-logits))
    
    # Convert to binary predictions (threshold = 0.5)
    predictions = (probabilities > 0.5).astype(int)
    
    return predictions.tolist()

def evaluate_classifier(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate key evaluation metrics.
    
    These are what interviewers ask about most:
    - Accuracy: Overall correctness
    - Precision: Of predicted positives, how many are correct?
    - Recall: Of actual positives, how many did we find?
    - F1: Balanced measure combining precision and recall
    """
    # Basic accuracy calculation
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true) if y_true else 0.0
    
    # For binary classification, calculate detailed metrics
    if set(y_true + y_pred) <= {0, 1}:
        # Confusion matrix components
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)  # True Positives
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)  # False Positives
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)  # False Negatives
        
        # Calculate metrics with zero-division protection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1
        }
    
    return {'accuracy': accuracy}

# COMPLETE INTERVIEW WALKTHROUGH
if __name__ == "__main__":
    print("TEXT CLASSIFICATION - Interview Walkthrough")
    print("=" * 50)
    
    # Sample data for interview demo
    train_texts = [
        "This movie is great!",         # Positive
        "Terrible film, hated it",      # Negative  
        "Amazing acting and plot",      # Positive
        "Boring and predictable",       # Negative
    ]
    train_labels = [1, 0, 1, 0]  # Binary: 1=positive, 0=negative
    
    print("TRAINING DATA:")
    for text, label in zip(train_texts, train_labels):
        sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
        print(f"  '{text}' -> {sentiment}")
    
    print(f"\n" + "STEP 1: FEATURE EXTRACTION")
    print("-" * 30)
    
    # Extract TF-IDF features
    X_train, vocabulary = extract_features(train_texts, method='tfidf')
    
    print(f"Vocabulary: {vocabulary}")
    print(f"Feature matrix: {len(X_train)} documents x {len(vocabulary)} features")
    
    # Show first document's features
    print(f"\nDocument 0 ('{train_texts[0]}') TF-IDF features:")
    for i, word in enumerate(vocabulary):
        score = X_train[0][i]
        if score > 0:
            print(f"  '{word}': {score:.3f}")
    
    print(f"\n" + "STEP 2: MODEL TRAINING")
    print("-" * 30)
    
    # Train model
    model = train_logistic_regression(X_train, train_labels)
    print("✓ Logistic regression trained")
    print(f"Model weights shape: {len(model['weights'])}")
    
    print(f"\n" + "STEP 3: PREDICTION")
    print("-" * 30)
    
    # Test on new examples
    test_texts = [
        "Excellent movie, loved every minute!",
        "Waste of time, very disappointing"
    ]
    
    # Extract features for test data
    # IMPORTANT: Use same vocabulary as training
    all_texts = train_texts + test_texts
    X_all, _ = extract_features(all_texts, method='tfidf')
    X_test = X_all[-len(test_texts):]  # Get test features
    
    # Make predictions
    predictions = predict_logistic(X_test, model)
    
    print("TEST RESULTS:")
    for text, pred in zip(test_texts, predictions):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"  '{text}' -> {sentiment}")
    
    print(f"\n" + "STEP 4: EVALUATION")
    print("-" * 30)
    
    # Evaluate on training data (normally you'd use test set)
    train_predictions = predict_logistic(X_train, model)
    metrics = evaluate_classifier(train_labels, train_predictions)
    
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.3f}")
    
    print(f"\n" + "=" * 50)
    print("KEY INTERVIEW INSIGHTS:")
    print("=" * 50)
    print("1. FEATURE ENGINEERING is crucial")
    print("   • TF-IDF captures word importance better than raw counts")
    print("   • Consider n-grams for phrase-level features")
    print("   • Word embeddings for semantic similarity")
    print()
    print("2. ALGORITHM CHOICE matters less than features")
    print("   • Logistic Regression: Fast, interpretable baseline")
    print("   • Naive Bayes: Good for small datasets")
    print("   • SVM: Good for high-dimensional sparse data")
    print()
    print("3. EVALUATION considerations")
    print("   • Use F1 score for imbalanced classes")
    print("   • Cross-validation for robust estimates")
    print("   • Precision vs Recall trade-off depends on use case")
    print()
    print("4. PRODUCTION challenges")
    print("   • Handle new vocabulary in test data")
    print("   • Monitor for data drift over time")
    print("   • Consider computational efficiency at scale")
