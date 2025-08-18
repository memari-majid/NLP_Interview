import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# PyTorch Implementation
if PYTORCH_AVAILABLE:
    
    class TextCNN(nn.Module):
        """1D CNN for text classification (Kim 2014 style)."""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 128,
                     num_classes: int = 2, filter_sizes: List[int] = [3, 4, 5],
                     num_filters: int = 100, dropout: float = 0.5):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            # Multiple convolution layers with different filter sizes
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
                for fs in filter_sizes
            ])
            
            self.dropout = nn.Dropout(dropout)
            
            # Fully connected layer
            self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
            
            # Store config
            self.filter_sizes = filter_sizes
            self.num_filters = num_filters
        
        def forward(self, x):
            # x shape: (batch_size, sequence_length)
            
            # Embedding
            embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
            embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
            
            # Apply convolutions
            conv_outputs = []
            for conv in self.convs:
                # Convolution + ReLU
                conv_out = F.relu(conv(embedded))  # (batch, num_filters, new_seq_len)
                
                # Max pooling over sequence
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
                pooled = pooled.squeeze(2)  # (batch, num_filters)
                
                conv_outputs.append(pooled)
            
            # Concatenate all conv outputs
            concatenated = torch.cat(conv_outputs, dim=1)  # (batch, len(filter_sizes) * num_filters)
            
            # Dropout
            dropped = self.dropout(concatenated)
            
            # Final classification
            output = self.fc(dropped)
            
            return output, concatenated  # Return both predictions and features
        
        def get_conv_features(self, x):
            """Extract convolutional features before pooling."""
            embedded = self.embedding(x)
            embedded = embedded.permute(0, 2, 1)
            
            conv_features = {}
            for i, conv in enumerate(self.convs):
                conv_out = F.relu(conv(embedded))
                conv_features[f'filter_size_{self.filter_sizes[i]}'] = conv_out
            
            return conv_features
    
    
    class CharacterCNN(nn.Module):
        """Character-level CNN for text classification."""
        
        def __init__(self, num_chars: int = 128, num_classes: int = 2,
                     conv_layers: List[Tuple[int, int]] = [(256, 7), (256, 3), (256, 3)],
                     fc_layers: List[int] = [1024, 1024], dropout: float = 0.5):
            super().__init__()
            
            # Character embedding (or one-hot)
            self.num_chars = num_chars
            
            # Convolutional layers
            self.convs = nn.ModuleList()
            in_channels = num_chars
            
            for out_channels, kernel_size in conv_layers:
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                        nn.ReLU(),
                        nn.MaxPool1d(2)
                    )
                )
                in_channels = out_channels
            
            # Calculate final conv output size (depends on input length)
            self.conv_output_dim = in_channels
            
            # Fully connected layers
            self.fc_layers = nn.ModuleList()
            in_features = self.conv_output_dim * 16  # Assuming pooled to ~16 dims
            
            for out_features in fc_layers:
                self.fc_layers.append(
                    nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
                in_features = out_features
            
            self.output = nn.Linear(in_features, num_classes)
        
        def forward(self, x):
            # x shape: (batch_size, sequence_length, num_chars) - one-hot encoded
            
            # Transpose for conv1d
            x = x.permute(0, 2, 1)  # (batch, num_chars, seq_len)
            
            # Apply convolutions
            for conv in self.convs:
                x = conv(x)
            
            # Global max pooling
            x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            
            # Fully connected layers
            for fc in self.fc_layers:
                x = fc(x)
            
            output = self.output(x)
            return output


# TensorFlow Implementation
if TENSORFLOW_AVAILABLE:
    
    def build_cnn_model_tf(vocab_size: int, embedding_dim: int = 128,
                          num_classes: int = 2, max_length: int = 100) -> keras.Model:
        """Build CNN model using TensorFlow/Keras."""
        
        inputs = layers.Input(shape=(max_length,))
        
        # Embedding layer
        embedded = layers.Embedding(vocab_size, embedding_dim, 
                                  mask_zero=True)(inputs)
        
        # Multiple conv layers with different filter sizes
        conv_outputs = []
        filter_sizes = [3, 4, 5]
        num_filters = 100
        
        for fs in filter_sizes:
            # 1D convolution
            conv = layers.Conv1D(num_filters, fs, activation='relu')(embedded)
            # Max pooling
            pool = layers.GlobalMaxPooling1D()(conv)
            conv_outputs.append(pool)
        
        # Concatenate all conv outputs
        concatenated = layers.concatenate(conv_outputs)
        
        # Dropout
        dropped = layers.Dropout(0.5)(concatenated)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(dropped)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        return model


# Common functions
def create_vocabulary(texts: List[str]) -> Dict[str, int]:
    """Create vocabulary from texts."""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    return vocab


def texts_to_sequences(texts: List[str], vocab: Dict[str, int], 
                      max_length: Optional[int] = None) -> List[List[int]]:
    """Convert texts to sequences."""
    sequences = []
    
    for text in texts:
        seq = [vocab.get(word.lower(), vocab['<UNK>']) for word in text.split()]
        
        if max_length:
            if len(seq) > max_length:
                seq = seq[:max_length]
            else:
                seq = seq + [vocab['<PAD>']] * (max_length - len(seq))
        
        sequences.append(seq)
    
    return sequences


def build_cnn_model(vocab_size: int, embedding_dim: int = 128,
                   num_classes: int = 2, framework: str = 'pytorch') -> nn.Module:
    """Build CNN model for text classification."""
    if framework == 'pytorch' and PYTORCH_AVAILABLE:
        return TextCNN(vocab_size, embedding_dim, num_classes)
    elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return build_cnn_model_tf(vocab_size, embedding_dim, num_classes)
    else:
        raise ValueError(f"Framework {framework} not available")


def train_cnn(model, texts: List[str], labels: List[int], 
             vocab: Dict[str, int], epochs: int = 10,
             batch_size: int = 32, learning_rate: float = 0.001):
    """Train CNN model."""
    if isinstance(model, TextCNN):
        return train_cnn_pytorch(model, texts, labels, vocab, epochs, 
                               batch_size, learning_rate)
    else:
        return train_cnn_tensorflow(model, texts, labels, vocab, epochs, 
                                  batch_size, learning_rate)


def train_cnn_pytorch(model: TextCNN, texts: List[str], labels: List[int],
                     vocab: Dict[str, int], epochs: int = 10,
                     batch_size: int = 32, learning_rate: float = 0.001):
    """Train PyTorch CNN model."""
    if not PYTORCH_AVAILABLE:
        return model
    
    # Convert texts to sequences
    sequences = texts_to_sequences(texts, vocab)
    
    # Create tensors
    X = [torch.tensor(seq) for seq in sequences]
    y = torch.tensor(labels)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Shuffle data
        indices = torch.randperm(len(X))
        
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = [X[j] for j in batch_indices]
            batch_y = y[batch_indices]
            
            # Pad sequences
            batch_X_padded = pad_sequence(batch_X, batch_first=True, padding_value=0)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _ = model(batch_X_padded)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / len(labels)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return model


def extract_features(model, text: str, vocab: Dict[str, int]) -> np.ndarray:
    """Extract CNN features from text."""
    if not PYTORCH_AVAILABLE:
        return np.array([])
    
    # Convert to sequence
    seq = texts_to_sequences([text], vocab)[0]
    x = torch.tensor([seq])
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, TextCNN):
            _, features = model(x)
            return features.numpy()[0]
    
    return np.array([])


def visualize_filters(model) -> Dict[str, np.ndarray]:
    """Visualize CNN filters."""
    if not PYTORCH_AVAILABLE or not isinstance(model, TextCNN):
        return {}
    
    filter_weights = {}
    
    for i, conv in enumerate(model.convs):
        # Get conv weights
        weights = conv.weight.data.cpu().numpy()
        filter_weights[f'filter_size_{model.filter_sizes[i]}'] = weights
    
    return filter_weights


def create_attention_cnn(vocab_size: int, embedding_dim: int = 128,
                        num_classes: int = 2) -> nn.Module:
    """CNN with attention mechanism."""
    if not PYTORCH_AVAILABLE:
        return None
    
    class AttentionCNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # CNN layers
            self.conv1 = nn.Conv1d(embedding_dim, 128, 3, padding=1)
            self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
            
            # Attention
            self.attention = nn.Linear(128, 1)
            
            # Classifier
            self.fc = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x).permute(0, 2, 1)
            
            # CNN
            conv1 = F.relu(self.conv1(embedded))
            conv2 = F.relu(self.conv2(conv1))
            
            # Attention
            conv2_t = conv2.permute(0, 2, 1)  # (batch, seq_len, channels)
            attention_scores = self.attention(conv2_t).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Weighted sum
            weighted = torch.bmm(conv2, attention_weights.unsqueeze(-1)).squeeze(-1)
            
            # Classification
            dropped = self.dropout(weighted)
            output = self.fc(dropped)
            
            return output, attention_weights
    
    return AttentionCNN(vocab_size, embedding_dim, num_classes)


# Demo data and functions
def create_movie_review_data():
    """Create sample movie review data."""
    positive_reviews = [
        "This movie is absolutely fantastic! Best film of the year.",
        "Amazing performances and brilliant storytelling. Loved every minute.",
        "A masterpiece of cinema. Highly recommend to everyone.",
        "Incredible movie with outstanding acting and direction.",
        "One of the best films I've ever seen. Simply amazing."
    ]
    
    negative_reviews = [
        "Terrible movie. Complete waste of time and money.",
        "Boring plot and awful acting. Very disappointing.",
        "One of the worst films ever made. Avoid at all costs.",
        "Poorly executed with no redeeming qualities whatsoever.",
        "Absolutely dreadful. I want my money back."
    ]
    
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    return texts, labels


def char_level_preprocessing(text: str, num_chars: int = 128) -> np.ndarray:
    """Convert text to character-level one-hot encoding."""
    # Create character to index mapping
    chars = list(set(text.lower()))[:num_chars-1]
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    char_to_idx['<UNK>'] = num_chars - 1
    
    # Convert text to indices
    indices = [char_to_idx.get(ch, char_to_idx['<UNK>']) for ch in text.lower()]
    
    # One-hot encode
    one_hot = np.zeros((len(indices), num_chars))
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1
    
    return one_hot


if __name__ == "__main__" and PYTORCH_AVAILABLE:
    print("CNN for Text Classification\n")
    
    # Create sample data
    texts, labels = create_movie_review_data()
    print(f"Dataset: {len(texts)} reviews\n")
    
    # Create vocabulary
    vocab = create_vocabulary(texts)
    print(f"Vocabulary size: {len(vocab)}\n")
    
    # Build and train word-level CNN
    print("Training Word-Level CNN...")
    model = build_cnn_model(len(vocab), embedding_dim=128, num_classes=2)
    model = train_cnn(model, texts, labels, vocab, epochs=5)
    
    print("\n" + "="*50 + "\n")
    
    # Test predictions
    test_texts = [
        "This is an amazing movie!",
        "Terrible film, very boring.",
        "Not bad but could be better."
    ]
    
    print("Test Predictions:")
    model.eval()
    for text in test_texts:
        features = extract_features(model, text, vocab)
        
        # Get prediction
        seq = texts_to_sequences([text], vocab)[0]
        x = torch.tensor([seq])
        
        with torch.no_grad():
            output, _ = model(x)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        sentiment = "positive" if pred_class == 1 else "negative"
        print(f"\nText: '{text}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        print(f"Feature vector shape: {features.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # Visualize filters
    print("Filter Visualization:")
    filter_weights = visualize_filters(model)
    for name, weights in filter_weights.items():
        print(f"{name}: shape = {weights.shape}")
        print(f"  First filter stats: mean={weights[0].mean():.3f}, std={weights[0].std():.3f}")
    
    print("\n" + "="*50 + "\n")
    
    # Character-level CNN example
    print("Character-Level CNN Example:")
    char_model = CharacterCNN(num_chars=70, num_classes=2)
    print(f"Model parameters: {sum(p.numel() for p in char_model.parameters()):,}")
    
    # Example input
    sample_text = "This movie is great!"
    char_input = char_level_preprocessing(sample_text, num_chars=70)
    print(f"Character input shape: {char_input.shape}")
    
    # Note: Full training would require more data and proper batching
