# Problem: Rule-based Sentiment Analysis

**Time: 20 minutes**

Implement a simple rule-based sentiment analyzer.

```python
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using rules (lexicon + modifiers).
    
    Returns:
        {"positive": 0.6, "negative": 0.1, "neutral": 0.3, "compound": 0.5}
    
    Rules:
    - Use sentiment lexicon for base scores
    - "very/really" intensifies sentiment (+30%)  
    - "not/never" flips sentiment (* -0.8)
    - Multiple punctuation adds emphasis
    """
    pass
```

**Requirements:**
- Create basic positive/negative word dictionary
- Handle intensifiers ("very good" -> higher positive score)
- Handle negations ("not bad" -> less negative)
- Normalize scores to sum to 1.0

**Follow-up:** How would you handle sarcasm or domain-specific sentiment?