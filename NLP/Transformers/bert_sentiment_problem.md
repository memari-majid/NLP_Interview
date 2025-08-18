# Problem: BERT Fine-tuning for Sentiment Analysis

Implement BERT fine-tuning for sentiment classification:
1. `load_pretrained_bert(model_name: str = 'bert-base-uncased') -> Model`
2. `fine_tune_bert(model, texts: List[str], labels: List[int], epochs: int = 3) -> Model`
3. `predict_with_bert(model, texts: List[str]) -> List[Tuple[str, float, Dict]]`

Example:
Input: "This product exceeded all my expectations!"
Output: ("positive", 0.98, {"attention_scores": [...], "cls_embedding": [...]})

Requirements:
- Use Hugging Face Transformers
- Handle tokenization with special tokens
- Implement proper fine-tuning strategy (freeze/unfreeze layers)
- Extract attention visualizations

Follow-ups:
- Compare BERT vs DistilBERT vs RoBERTa
- Multi-class classification
- Few-shot learning with prompts
