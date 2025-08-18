from typing import Dict, List

# Basic sentiment lexicon  
SENTIMENT_LEXICON = {
    'good': 2.0, 'great': 3.0, 'excellent': 3.5, 'amazing': 3.5, 'love': 3.0,
    'like': 2.0, 'happy': 2.5, 'fantastic': 3.5, 'wonderful': 3.0,
    'bad': -2.0, 'terrible': -3.0, 'awful': -3.5, 'hate': -3.0, 'sad': -2.5,
    'angry': -2.5, 'disappointing': -2.5, 'worst': -3.5, 'horrible': -3.5
}

INTENSIFIERS = {'very', 'really', 'extremely', 'incredibly', 'absolutely'}
NEGATIONS = {'not', 'never', 'no', 'nothing', 'nowhere', 'nobody'}

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment using lexicon and rules."""
    if not text:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    words = text.lower().split()
    scores = []
    
    i = 0
    while i < len(words):
        word = words[i]
        
        if word in SENTIMENT_LEXICON:
            score = SENTIMENT_LEXICON[word]
            
            # Check for intensifier in previous word
            if i > 0 and words[i-1] in INTENSIFIERS:
                score *= 1.3  # Boost by 30%
            
            # Check for negation in previous 2 words
            negated = False
            for j in range(max(0, i-2), i):
                if words[j] in NEGATIONS:
                    negated = True
                    break
            
            if negated:
                score *= -0.8  # Flip and reduce intensity
            
            scores.append(score)
        
        i += 1
    
    # Add punctuation emphasis
    exclamation_count = text.count('!')
    if exclamation_count > 0 and scores:
        emphasis = min(exclamation_count * 0.3, 1.0)  # Cap emphasis
        scores = [s * (1 + emphasis) if s > 0 else s * (1 + emphasis) for s in scores]
    
    # Calculate final scores
    if not scores:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    # Separate positive and negative
    pos_scores = [s for s in scores if s > 0]
    neg_scores = [s for s in scores if s < 0]
    
    pos_sum = sum(pos_scores)
    neg_sum = abs(sum(neg_scores))
    
    # Normalize to proportions
    total = pos_sum + neg_sum
    if total > 0:
        pos_prop = pos_sum / total
        neg_prop = neg_sum / total
        neu_prop = 0.0
    else:
        pos_prop = neg_prop = 0.0
        neu_prop = 1.0
    
    # Compound score (overall sentiment)
    compound = (pos_sum - neg_sum) / (total + 1) if total > 0 else 0.0
    compound = max(-1, min(1, compound))  # Clamp to [-1, 1]
    
    return {
        "positive": round(pos_prop, 3),
        "negative": round(neg_prop, 3),
        "neutral": round(neu_prop, 3),
        "compound": round(compound, 3)
    }

def classify_sentiment(sentiment_scores: Dict[str, float]) -> str:
    """Classify overall sentiment from scores."""
    compound = sentiment_scores['compound']
    
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Test
if __name__ == "__main__":
    test_cases = [
        "This movie is great!",
        "I hate this terrible product",  
        "Not bad at all",
        "Very good movie",
        "This is okay",
        "Really amazing performance!!!"
    ]
    
    for text in test_cases:
        scores = analyze_sentiment(text)
        classification = classify_sentiment(scores)
        print(f"'{text}' -> {classification} (compound: {scores['compound']})")
        print(f"  Breakdown: pos={scores['positive']}, neg={scores['negative']}, neu={scores['neutral']}")
        print()