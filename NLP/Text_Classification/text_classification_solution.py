import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


class TextClassificationModel:
    """Wrapper for text classification models."""
    
    def __init__(self, vectorizer, classifier, label_encoder, confidence_threshold=0.7):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.confidence_threshold = confidence_threshold
        self.feature_names = None
    
    def predict(self, texts: List[str]) -> List[str]:
        """Predict labels for texts."""
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred).tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for each class."""
        X = self.vectorizer.transform(texts)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        else:
            # For SVM, use decision function
            decision = self.classifier.decision_function(X)
            if len(self.label_encoder.classes_) == 2:
                # Binary classification
                proba = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - proba, proba])
            else:
                # Multi-class: softmax on decision values
                exp_decision = np.exp(decision)
                return exp_decision / exp_decision.sum(axis=1, keepdims=True)
    
    def predict_with_confidence(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict with confidence scores."""
        probas = self.predict_proba(texts)
        predictions = []
        
        for i, proba in enumerate(probas):
            max_idx = np.argmax(proba)
            label = self.label_encoder.inverse_transform([max_idx])[0]
            confidence = proba[max_idx]
            predictions.append((label, confidence))
        
        return predictions
    
    def get_uncertain_predictions(self, texts: List[str]) -> List[Tuple[int, str, float]]:
        """Get predictions with low confidence for active learning."""
        predictions_with_conf = self.predict_with_confidence(texts)
        uncertain = []
        
        for i, (label, conf) in enumerate(predictions_with_conf):
            if conf < self.confidence_threshold:
                uncertain.append((i, texts[i], conf))
        
        return uncertain


def train_classifier(texts: List[str], labels: List[str], 
                    algorithm: str = 'logistic_regression',
                    handle_imbalance: bool = True) -> TextClassificationModel:
    """Train a text classifier."""
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Select classifier
    if algorithm == 'logistic_regression':
        if handle_imbalance:
            classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
        else:
            classifier = LogisticRegression(max_iter=1000)
    elif algorithm == 'naive_bayes':
        classifier = MultinomialNB(alpha=0.1)
    elif algorithm == 'svm':
        if handle_imbalance:
            classifier = LinearSVC(class_weight='balanced', max_iter=1000)
        else:
            classifier = LinearSVC(max_iter=1000)
    elif algorithm == 'random_forest':
        if handle_imbalance:
            classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        else:
            classifier = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create pipeline and train
    X = vectorizer.fit_transform(texts)
    classifier.fit(X, y)
    
    # Create model wrapper
    model = TextClassificationModel(vectorizer, classifier, label_encoder)
    model.feature_names = vectorizer.get_feature_names_out()
    
    return model


def evaluate_classifier(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Evaluate classifier performance."""
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    labels = sorted(set(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    results = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_metrics': {
            label: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            } for i, label in enumerate(labels)
        },
        'confusion_matrix': cm.tolist(),
        'labels': labels
    }
    
    return results


def get_feature_importance(model: TextClassificationModel, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    """Extract feature importance for interpretability."""
    feature_importance = {}
    
    if hasattr(model.classifier, 'coef_'):  # Linear models
        # For each class
        for i, label in enumerate(model.label_encoder.classes_):
            if len(model.label_encoder.classes_) == 2 and i > 0:
                # Binary classification: only one set of coefficients
                coef = model.classifier.coef_[0]
            else:
                coef = model.classifier.coef_[i]
            
            # Get top positive and negative features
            top_positive_idx = np.argsort(coef)[-top_k:][::-1]
            top_negative_idx = np.argsort(coef)[:top_k]
            
            positive_features = [(model.feature_names[idx], coef[idx]) 
                               for idx in top_positive_idx]
            negative_features = [(model.feature_names[idx], coef[idx]) 
                               for idx in top_negative_idx]
            
            feature_importance[label] = {
                'positive': positive_features,
                'negative': negative_features
            }
    
    elif hasattr(model.classifier, 'feature_importances_'):  # Tree-based models
        importances = model.classifier.feature_importances_
        top_idx = np.argsort(importances)[-top_k:][::-1]
        
        feature_importance['overall'] = [
            (model.feature_names[idx], importances[idx]) 
            for idx in top_idx
        ]
    
    return feature_importance


def cross_validate_classifier(texts: List[str], labels: List[str], 
                            algorithms: List[str] = None,
                            cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
    """Compare multiple classifiers using cross-validation."""
    if algorithms is None:
        algorithms = ['logistic_regression', 'naive_bayes', 'svm', 'random_forest']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Vectorize texts
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    
    results = {}
    
    for algo in algorithms:
        # Create classifier
        if algo == 'logistic_regression':
            clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        elif algo == 'naive_bayes':
            clf = MultinomialNB(alpha=0.1)
        elif algo == 'svm':
            clf = LinearSVC(class_weight='balanced', max_iter=1000)
        elif algo == 'random_forest':
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        
        # Cross-validation
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='f1_macro')
        
        results[algo] = {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores.tolist()
        }
    
    return results


class MultiLabelTextClassifier:
    """Multi-label text classification (texts can have multiple labels)."""
    
    def __init__(self, algorithm='logistic_regression'):
        self.algorithm = algorithm
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifiers = {}
        self.labels = []
    
    def train(self, texts: List[str], labels_list: List[List[str]]):
        """Train multi-label classifier.
        
        Args:
            texts: List of texts
            labels_list: List of label lists (each text can have multiple labels)
        """
        # Get unique labels
        all_labels = set()
        for labels in labels_list:
            all_labels.update(labels)
        self.labels = sorted(all_labels)
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train binary classifier for each label
        for label in self.labels:
            # Create binary labels
            y_binary = np.array([1 if label in labels else 0 
                               for labels in labels_list])
            
            # Train classifier
            if self.algorithm == 'logistic_regression':
                clf = LogisticRegression(class_weight='balanced', max_iter=1000)
            else:
                clf = LinearSVC(class_weight='balanced', max_iter=1000)
            
            clf.fit(X, y_binary)
            self.classifiers[label] = clf
    
    def predict(self, texts: List[str], threshold: float = 0.5) -> List[List[str]]:
        """Predict labels for texts."""
        X = self.vectorizer.transform(texts)
        predictions = []
        
        for i in range(X.shape[0]):
            text_labels = []
            
            for label in self.labels:
                clf = self.classifiers[label]
                
                if hasattr(clf, 'predict_proba'):
                    prob = clf.predict_proba(X[i])[0][1]
                else:
                    # For SVM, use decision function
                    decision = clf.decision_function(X[i])[0]
                    prob = 1 / (1 + np.exp(-decision))
                
                if prob >= threshold:
                    text_labels.append(label)
            
            predictions.append(text_labels)
        
        return predictions


if __name__ == "__main__":
    # Example 1: Basic text classification
    print("Example 1: Sentiment Classification")
    
    # Training data
    train_texts = [
        "This movie is fantastic! Best I've seen all year.",
        "Terrible experience, would not recommend to anyone.",
        "It was okay, nothing special but not bad either.",
        "Absolutely loved it! Amazing performances.",
        "Waste of time and money. Very disappointing.",
        "Average movie, some good parts but overall mediocre.",
        "Outstanding! A masterpiece of cinema.",
        "Boring and predictable. Fell asleep halfway through.",
        "Not bad, worth watching once."
    ]
    
    train_labels = [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive", "negative", "neutral"
    ]
    
    # Train classifier
    model = train_classifier(train_texts, train_labels)
    
    # Test predictions
    test_texts = [
        "Amazing film, loved every minute!",
        "Complete disaster, worst movie ever.",
        "It's fine, nothing to write home about."
    ]
    
    predictions = model.predict(test_texts)
    predictions_with_conf = model.predict_with_confidence(test_texts)
    
    print("\nPredictions:")
    for text, pred, (_, conf) in zip(test_texts, predictions, predictions_with_conf):
        print(f"Text: '{text}'")
        print(f"Prediction: {pred} (confidence: {conf:.3f})\n")
    
    print("="*50 + "\n")
    
    # Example 2: Evaluate different algorithms
    print("Example 2: Algorithm Comparison")
    
    # More data for better comparison
    from sklearn.datasets import fetch_20newsgroups
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    
    try:
        newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                                       shuffle=True, random_state=42)
        texts = newsgroups.data[:200]  # Use subset for speed
        labels = [categories[i] for i in newsgroups.target[:200]]
        
        cv_results = cross_validate_classifier(texts, labels, cv_folds=3)
        
        print("\nCross-validation results:")
        for algo, results in cv_results.items():
            print(f"{algo}: F1={results['mean_f1']:.3f} (+/- {results['std_f1']:.3f})")
        
    except Exception as e:
        print("Skipping newsgroups example:", e)
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Feature importance
    print("Example 3: Feature Importance Analysis")
    
    feature_importance = get_feature_importance(model, top_k=5)
    
    if feature_importance:
        for label, features in feature_importance.items():
            if isinstance(features, dict):
                print(f"\nClass '{label}':")
                print("  Top positive features:", features['positive'][:3])
                print("  Top negative features:", features['negative'][:3])
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Multi-label classification
    print("Example 4: Multi-label Classification")
    
    # Multi-label training data
    ml_texts = [
        "Python is great for machine learning and web development",
        "JavaScript is essential for frontend development",
        "Machine learning requires good math skills",
        "Web development with React and Node.js",
        "Deep learning with TensorFlow and PyTorch",
        "Data science involves statistics and programming"
    ]
    
    ml_labels = [
        ["programming", "machine learning", "web development"],
        ["programming", "web development"],
        ["machine learning", "education"],
        ["web development", "programming"],
        ["machine learning", "programming"],
        ["data science", "programming", "education"]
    ]
    
    # Train multi-label classifier
    ml_classifier = MultiLabelTextClassifier()
    ml_classifier.train(ml_texts, ml_labels)
    
    # Test
    test_ml_texts = [
        "Building neural networks with Python",
        "Creating responsive websites with CSS",
        "Statistical analysis and data visualization"
    ]
    
    ml_predictions = ml_classifier.predict(test_ml_texts)
    
    print("\nMulti-label predictions:")
    for text, labels in zip(test_ml_texts, ml_predictions):
        print(f"Text: '{text}'")
        print(f"Labels: {labels}\n")
