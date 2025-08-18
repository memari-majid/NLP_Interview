import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning frameworks
FRAMEWORK = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
    FRAMEWORK = 'pytorch'
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        FRAMEWORK = 'tensorflow'
    except ImportError:
        print("Please install PyTorch or TensorFlow")
        print("pip install torch or pip install tensorflow")


# Common preprocessing functions
def create_vocabulary(texts: List[str], max_vocab_size: int = 10000) -> Dict[str, int]:
    """Create vocabulary from texts."""
    from collections import Counter
    
    # Tokenize and count words
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Create vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    # Add most common words
    for word, _ in word_counts.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)
    
    return vocab


def texts_to_sequences(texts: List[str], vocab: Dict[str, int], max_len: Optional[int] = None) -> List[List[int]]:
    """Convert texts to sequences of integers."""
    sequences = []
    
    for text in texts:
        words = text.lower().split()
        sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
        
        if max_len:
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
            else:
                sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))
        
        sequences.append(sequence)
    
    return sequences


# PyTorch Implementation
if FRAMEWORK == 'pytorch':
    
    class LSTMSentimentClassifier(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int = 100, 
                     hidden_dim: int = 128, num_layers: int = 2,
                     dropout: float = 0.5, bidirectional: bool = True):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)
            
            self.lstm = nn.LSTM(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
            
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            
            # Attention layer
            self.attention = nn.Linear(lstm_output_dim, 1)
            
            # Output layer
            self.fc = nn.Linear(lstm_output_dim, 2)  # Binary classification
            
        def forward(self, x, lengths=None):
            # x shape: (batch_size, seq_len)
            embedded = self.dropout(self.embedding(x))
            
            if lengths is not None:
                # Pack sequences for efficiency
                packed = pack_padded_sequence(embedded, lengths.cpu(), 
                                            batch_first=True, enforce_sorted=False)
                lstm_out, (hidden, cell) = self.lstm(packed)
                lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            else:
                lstm_out, (hidden, cell) = self.lstm(embedded)
            
            # Apply attention
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            attended = torch.sum(attention_weights * lstm_out, dim=1)
            
            # Classification
            output = self.fc(attended)
            
            return output, attention_weights
    
    
    class GRUSentimentClassifier(nn.Module):
        """GRU variant for comparison."""
        def __init__(self, vocab_size: int, embedding_dim: int = 100,
                     hidden_dim: int = 128, num_layers: int = 2):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                             dropout=0.5 if num_layers > 1 else 0,
                             bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_dim * 2, 2)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x, lengths=None):
            embedded = self.dropout(self.embedding(x))
            _, hidden = self.gru(embedded)
            
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            output = self.fc(hidden)
            
            return output, None
    
    
    def build_lstm_model(vocab_size: int, embedding_dim: int = 100, 
                        model_type: str = 'lstm') -> nn.Module:
        """Build LSTM or GRU model."""
        if model_type == 'lstm':
            return LSTMSentimentClassifier(vocab_size, embedding_dim)
        elif model_type == 'gru':
            return GRUSentimentClassifier(vocab_size, embedding_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    
    def train_sentiment_model(texts: List[str], labels: List[int], 
                            vocab: Dict[str, int], epochs: int = 10,
                            batch_size: int = 32) -> nn.Module:
        """Train sentiment model."""
        # Convert texts to sequences
        sequences = texts_to_sequences(texts, vocab)
        
        # Create model
        model = build_lstm_model(len(vocab))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to tensors
        X = [torch.tensor(seq) for seq in sequences]
        y = torch.tensor(labels)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            # Mini-batch training
            indices = torch.randperm(len(X))
            
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = [X[j] for j in batch_indices]
                batch_y = y[batch_indices]
                
                # Pad sequences in batch
                lengths = torch.tensor([len(seq) for seq in batch_X])
                batch_X_padded = pad_sequence(batch_X, batch_first=True, padding_value=0)
                
                # Forward pass
                optimizer.zero_grad()
                output, _ = model(batch_X_padded, lengths)
                loss = criterion(output, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        return model
    
    
    def predict_sentiment(model: nn.Module, texts: List[str], 
                         vocab: Dict[str, int]) -> List[Tuple[int, float]]:
        """Predict sentiment with confidence."""
        model.eval()
        sequences = texts_to_sequences(texts, vocab)
        predictions = []
        
        with torch.no_grad():
            for seq in sequences:
                x = torch.tensor([seq])
                output, attention = model(x)
                probs = torch.softmax(output, dim=1)
                
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                
                predictions.append((pred_class, confidence))
        
        return predictions


# TensorFlow Implementation
elif FRAMEWORK == 'tensorflow':
    
    def build_lstm_model(vocab_size: int, embedding_dim: int = 100,
                        model_type: str = 'lstm') -> keras.Model:
        """Build LSTM model using Keras."""
        model = keras.Sequential()
        
        # Embedding layer
        model.add(layers.Embedding(vocab_size, embedding_dim, mask_zero=True))
        model.add(layers.Dropout(0.5))
        
        # RNN layer
        if model_type == 'lstm':
            model.add(layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, dropout=0.5)
            ))
            model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.5)))
        elif model_type == 'gru':
            model.add(layers.Bidirectional(
                layers.GRU(128, return_sequences=True, dropout=0.5)
            ))
            model.add(layers.Bidirectional(layers.GRU(64, dropout=0.5)))
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    
    def train_sentiment_model(texts: List[str], labels: List[int],
                            vocab: Dict[str, int], epochs: int = 10,
                            batch_size: int = 32) -> keras.Model:
        """Train sentiment model."""
        # Convert texts to sequences
        sequences = texts_to_sequences(texts, vocab, max_len=100)
        X = np.array(sequences)
        y = np.array(labels)
        
        # Create and train model
        model = build_lstm_model(len(vocab))
        
        # Use validation split
        model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        return model
    
    
    def predict_sentiment(model: keras.Model, texts: List[str],
                         vocab: Dict[str, int]) -> List[Tuple[int, float]]:
        """Predict sentiment with confidence."""
        sequences = texts_to_sequences(texts, vocab, max_len=100)
        X = np.array(sequences)
        
        probs = model.predict(X)
        predictions = []
        
        for prob in probs:
            pred_class = np.argmax(prob)
            confidence = prob[pred_class]
            predictions.append((pred_class, confidence))
        
        return predictions


# Common evaluation functions
def evaluate_model(model, texts: List[str], labels: List[int], 
                  vocab: Dict[str, int]) -> Dict[str, float]:
    """Evaluate model performance."""
    predictions = predict_sentiment(model, texts, vocab)
    
    y_pred = [pred[0] for pred in predictions]
    y_true = labels
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    accuracy = correct / len(y_true)
    
    # Calculate per-class metrics
    tp = [0, 0]  # True positives for each class
    fp = [0, 0]  # False positives
    fn = [0, 0]  # False negatives
    
    for pred, true in zip(y_pred, y_true):
        if pred == true:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    
    precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(2)]
    recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(2)]
    f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_f1': sum(f1) / len(f1)
    }


# Demo functions
def create_sample_data() -> Tuple[List[str], List[int], List[str], List[int]]:
    """Create sample sentiment data."""
    train_texts = [
        "This movie is absolutely fantastic! Best I've seen all year.",
        "Terrible experience, would not recommend to anyone.",
        "Amazing performances and great storyline. Loved it!",
        "Boring and predictable. Complete waste of time.",
        "One of the best films ever made. Truly exceptional.",
        "Awful movie. Bad acting and terrible plot.",
        "Incredible cinematography and outstanding performances.",
        "Very disappointing. Expected much better.",
        "A masterpiece! Everyone should watch this.",
        "Worst movie I've ever seen. Absolutely horrible."
    ]
    
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    
    test_texts = [
        "Fantastic movie! Highly recommend it.",
        "Not worth watching. Very bad.",
        "Decent film, nothing special."
    ]
    
    test_labels = [1, 0, 1]  # Last one is actually neutral but we'll treat as positive
    
    return train_texts, train_labels, test_texts, test_labels


if __name__ == "__main__" and FRAMEWORK:
    print(f"Using {FRAMEWORK} for implementation\n")
    
    # Create sample data
    train_texts, train_labels, test_texts, test_labels = create_sample_data()
    
    # Create vocabulary
    vocab = create_vocabulary(train_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Train model
    print("\nTraining LSTM model...")
    model = train_sentiment_model(train_texts, train_labels, vocab, epochs=10)
    
    # Make predictions
    print("\nMaking predictions on test set:")
    predictions = predict_sentiment(model, test_texts, vocab)
    
    for text, (pred_class, confidence), true_label in zip(test_texts, predictions, test_labels):
        sentiment = "positive" if pred_class == 1 else "negative"
        print(f"\nText: '{text}'")
        print(f"Predicted: {sentiment} (confidence: {confidence:.3f})")
        print(f"Actual: {'positive' if true_label == 1 else 'negative'}")
    
    # Evaluate model
    print("\n" + "="*50)
    print("Model evaluation on training set:")
    train_metrics = evaluate_model(model, train_texts, train_labels, vocab)
    print(f"Accuracy: {train_metrics['accuracy']:.3f}")
    print(f"Average F1: {train_metrics['avg_f1']:.3f}")
    
    # Compare with GRU
    if FRAMEWORK == 'pytorch':
        print("\n" + "="*50)
        print("Training GRU model for comparison...")
        gru_model = build_lstm_model(len(vocab), model_type='gru')
        # Note: In practice, you'd properly train the GRU model
        print("GRU model created (training skipped for brevity)")
    
    # Demonstrate attention weights (PyTorch only)
    if FRAMEWORK == 'pytorch' and hasattr(model, 'attention'):
        print("\n" + "="*50)
        print("Attention weights visualization (first test example):")
        
        model.eval()
        seq = texts_to_sequences([test_texts[0]], vocab)[0]
        x = torch.tensor([seq])
        
        with torch.no_grad():
            output, attention_weights = model(x)
            
        words = test_texts[0].split()
        attention = attention_weights[0, :len(words), 0].numpy()
        
        print(f"\nText: '{test_texts[0]}'")
        print("Word attention scores:")
        for word, score in zip(words, attention):
            print(f"  {word}: {'*' * int(score * 20)} ({score:.3f})")
