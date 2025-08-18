import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        BertTokenizer, BertForSequenceClassification, BertModel,
        DistilBertTokenizer, DistilBertForSequenceClassification,
        RobertaTokenizer, RobertaForSequenceClassification,
        AdamW, get_linear_schedule_with_warmup,
        AutoTokenizer, AutoModelForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Please install transformers: pip install transformers")


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTSentimentClassifier(nn.Module):
    """BERT-based sentiment classifier with additional features."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2, 
                 dropout: float = 0.3, freeze_bert: bool = False):
        super().__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits, outputs.attentions, pooled_output


def load_pretrained_bert(model_name: str = 'bert-base-uncased', 
                        task: str = 'custom') -> Tuple[Union[nn.Module, any], any]:
    """Load pre-trained BERT model and tokenizer."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    if task == 'custom':
        # Custom model with our classification head
        model = BERTSentimentClassifier(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        # Hugging Face model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def fine_tune_bert(model: nn.Module, texts: List[str], labels: List[int],
                  tokenizer, epochs: int = 3, batch_size: int = 16,
                  learning_rate: float = 2e-5, warmup_steps: int = 0) -> nn.Module:
    """Fine-tune BERT for sentiment analysis."""
    if not TRANSFORMERS_AVAILABLE:
        return model
    
    # Create dataset and dataloader
    dataset = SentimentDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if isinstance(model, BERTSentimentClassifier):
                logits, _, _ = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
            
            if isinstance(model, BERTSentimentClassifier):
                loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
    
    return model


def predict_with_bert(model: nn.Module, texts: List[str], tokenizer,
                     return_attention: bool = True) -> List[Tuple[str, float, Dict]]:
    """Make predictions with BERT model."""
    if not TRANSFORMERS_AVAILABLE:
        return []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get predictions
            if isinstance(model, BERTSentimentClassifier):
                logits, attentions, cls_embedding = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
                cls_embedding = outputs.hidden_states[-1][:, 0, :] if hasattr(outputs, 'hidden_states') else None
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_class].item()
            
            # Sentiment label
            sentiment = "positive" if predicted_class == 1 else "negative"
            
            # Additional information
            extra_info = {}
            
            if return_attention and attentions is not None:
                # Average attention across all layers and heads
                avg_attention = torch.stack(attentions).mean(dim=(0, 1))[0]
                extra_info['attention_scores'] = avg_attention.cpu().numpy().tolist()
            
            if cls_embedding is not None:
                extra_info['cls_embedding'] = cls_embedding[0].cpu().numpy().tolist()[:10]  # First 10 dims
            
            predictions.append((sentiment, confidence, extra_info))
    
    return predictions


def compare_transformer_models(texts: List[str], labels: List[int],
                             models: List[str] = None) -> Dict[str, Dict]:
    """Compare different transformer models."""
    if not TRANSFORMERS_AVAILABLE:
        return {}
    
    if models is None:
        models = ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']
    
    results = {}
    
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Load model and tokenizer
            if 'bert' in model_name and 'distil' not in model_name:
                tokenizer = BertTokenizer.from_pretrained(model_name)
                model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            elif 'distilbert' in model_name:
                tokenizer = DistilBertTokenizer.from_pretrained(model_name)
                model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            elif 'roberta' in model_name:
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results[model_name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
            }
            
            print(f"Total parameters: {total_params:,}")
            print(f"Model size: {results[model_name]['model_size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def few_shot_sentiment(model, tokenizer, examples: List[Tuple[str, str]], 
                      query: str) -> Tuple[str, float]:
    """Few-shot learning with prompts (for GPT-style models)."""
    # Create prompt
    prompt = "Classify the sentiment of the following texts:\n\n"
    
    for text, label in examples:
        prompt += f"Text: {text}\nSentiment: {label}\n\n"
    
    prompt += f"Text: {query}\nSentiment:"
    
    # This is a simplified version - in practice, you'd use a generative model
    # For BERT, we'll just use the fine-tuned classifier
    predictions = predict_with_bert(model, [query], tokenizer, return_attention=False)
    
    if predictions:
        sentiment, confidence, _ = predictions[0]
        return sentiment, confidence
    
    return "unknown", 0.0


def visualize_attention(text: str, tokens: List[str], attention_scores: List[float]):
    """Simple text-based attention visualization."""
    print(f"\nText: '{text}'")
    print("\nToken attention scores:")
    
    max_score = max(attention_scores) if attention_scores else 1
    
    for token, score in zip(tokens, attention_scores):
        # Normalize to 0-20 scale for visualization
        bar_length = int((score / max_score) * 20)
        bar = '█' * bar_length
        print(f"{token:15} {bar} {score:.3f}")


# Demo functions
def create_imdb_sample_data() -> Tuple[List[str], List[int], List[str], List[int]]:
    """Create sample IMDB-style movie review data."""
    train_texts = [
        "This movie is a masterpiece. The acting is superb and the story is captivating.",
        "Terrible film. Waste of time and money. Poor acting and boring plot.",
        "Absolutely loved it! One of the best movies I've seen this year.",
        "Disappointing. Had high expectations but the movie failed to deliver.",
        "Brilliant cinematography and outstanding performances by all actors.",
        "Couldn't even finish watching it. Extremely boring and predictable.",
        "A true work of art. Every scene is beautifully crafted.",
        "Not worth the hype. Mediocre at best.",
        "Exceptional movie that will be remembered for years to come.",
        "One of the worst movies ever made. Avoid at all costs."
    ]
    
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    test_texts = [
        "Amazing film with incredible performances!",
        "Boring and poorly executed. Not recommended.",
        "Decent movie but nothing groundbreaking."
    ]
    
    test_labels = [1, 0, 1]
    
    return train_texts, train_labels, test_texts, test_labels


if __name__ == "__main__" and TRANSFORMERS_AVAILABLE:
    print("BERT Fine-tuning for Sentiment Analysis\n")
    
    # Create sample data
    train_texts, train_labels, test_texts, test_labels = create_imdb_sample_data()
    
    # Load pre-trained BERT
    print("Loading pre-trained BERT...")
    model, tokenizer = load_pretrained_bert('bert-base-uncased', task='custom')
    
    if model and tokenizer:
        # Fine-tune on sample data
        print("\nFine-tuning BERT (this will take a moment)...")
        model = fine_tune_bert(
            model, train_texts, train_labels, tokenizer, 
            epochs=2, batch_size=4, learning_rate=2e-5
        )
        
        # Make predictions
        print("\nMaking predictions on test set:")
        predictions = predict_with_bert(model, test_texts, tokenizer)
        
        for text, (sentiment, confidence, extra_info) in zip(test_texts, predictions):
            print(f"\nText: '{text}'")
            print(f"Predicted: {sentiment} (confidence: {confidence:.3f})")
            
            # Show attention visualization if available
            if 'attention_scores' in extra_info:
                tokens = tokenizer.tokenize(text)[:20]  # First 20 tokens
                attention = extra_info['attention_scores'][:20]
                visualize_attention(text, tokens, attention)
        
        print("\n" + "="*50 + "\n")
        
        # Compare different models
        print("Comparing transformer models:")
        model_comparison = compare_transformer_models(
            train_texts[:2], train_labels[:2],
            models=['bert-base-uncased', 'distilbert-base-uncased']
        )
        
        print("\nModel comparison summary:")
        for model_name, info in model_comparison.items():
            if 'error' not in info:
                print(f"{model_name}: {info['model_size_mb']:.2f} MB")
        
        print("\n" + "="*50 + "\n")
        
        # Few-shot example
        print("Few-shot learning example:")
        few_shot_examples = [
            ("The food was delicious", "positive"),
            ("Service was terrible", "negative")
        ]
        
        query = "The meal was fantastic and the ambiance was perfect"
        sentiment, confidence = few_shot_sentiment(model, tokenizer, few_shot_examples, query)
        print(f"\nQuery: '{query}'")
        print(f"Predicted: {sentiment} (confidence: {confidence:.3f})")
