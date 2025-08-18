from typing import Dict, List

# SENTIMENT LEXICON: Core vocabulary with polarity scores
# In interviews, mention this would be much larger in production (thousands of words)
# Scores range from -4 (very negative) to +4 (very positive)
SENTIMENT_LEXICON = {
    # Positive words (common in interviews)
    'good': 2.0, 'great': 3.0, 'excellent': 3.5, 'amazing': 3.5, 'love': 3.0,
    'like': 2.0, 'happy': 2.5, 'fantastic': 3.5, 'wonderful': 3.0, 'awesome': 3.5,
    
    # Negative words (common in interviews)  
    'bad': -2.0, 'terrible': -3.0, 'awful': -3.5, 'hate': -3.0, 'sad': -2.5,
    'angry': -2.5, 'disappointing': -2.5, 'worst': -3.5, 'horrible': -3.5
}

# INTENSIFIERS: Words that boost sentiment strength
# "very good" should be more positive than just "good"
INTENSIFIERS = {'very', 'really', 'extremely', 'incredibly', 'absolutely'}

# NEGATIONS: Words that flip sentiment polarity  
# "not good" should be negative, not positive
NEGATIONS = {'not', 'never', 'no', 'nothing', 'nowhere', 'nobody', 'cannot', "can't", "don't", "isn't", "wasn't", "won't", "wouldn't", "shouldn't", "couldn't"}

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using rule-based approach (VADER-style).
    
    RULE-BASED SENTIMENT ANALYSIS:
    - Uses pre-built dictionary of word sentiments
    - Applies grammatical rules (intensifiers, negations)
    - Fast and interpretable (good for production)
    - No training data needed
    
    ALGORITHM:
    1. Tokenize text and look up word sentiments
    2. Apply intensifier rules ("very good" > "good")
    3. Apply negation rules ("not good" becomes negative)
    4. Handle punctuation emphasis ("great!!!" > "great")
    5. Normalize scores and return distribution
    """
    
    # STEP 1: Handle edge cases first
    if not text or not text.strip():
        # Return neutral sentiment for empty text
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    # STEP 2: Simple tokenization
    # Split on whitespace and convert to lowercase
    words = text.lower().split()
    
    if not words:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    # STEP 3: Get base sentiment scores for each word
    # Look up each word in sentiment lexicon
    scores = []
    
    for i, word in enumerate(words):
        # Get base sentiment score (0 if not in lexicon)
        base_score = SENTIMENT_LEXICON.get(word, 0.0)
        
        if base_score != 0:  # Only process sentiment-bearing words
            
            # RULE 1: Check for intensifiers in previous word
            # "very good" should be more positive than "good"
            if i > 0 and words[i-1] in INTENSIFIERS:
                # Boost sentiment by 30% (VADER-style boosting)
                base_score *= 1.3
                print(f"  INTENSIFIER: '{words[i-1]} {word}' -> boosted to {base_score:.2f}")
            
            # RULE 2: Check for negations in previous 2 words
            # "not very good" should be negative despite "good" being positive
            negated = False
            for j in range(max(0, i-2), i):  # Look back up to 2 words
                if words[j] in NEGATIONS:
                    negated = True
                    print(f"  NEGATION: '{words[j]}' flips '{word}'")
                    break
            
            # Apply negation: flip polarity and reduce intensity
            if negated:
                base_score *= -0.8  # Flip sign and dampen (VADER approach)
            
            scores.append(base_score)
    
    # STEP 4: Handle punctuation emphasis
    # Multiple exclamation marks add emphasis: "Great!!!" > "Great"
    exclamation_count = text.count('!')
    if exclamation_count > 0 and scores:
        # Add emphasis but cap the effect
        emphasis = min(exclamation_count * 0.3, 1.0)
        print(f"  EMPHASIS: {exclamation_count} exclamations add {emphasis:.2f}")
        
        # Apply emphasis to existing sentiment
        scores = [s * (1 + emphasis) if s > 0 else s * (1 + emphasis) for s in scores]
    
    # STEP 5: Calculate final sentiment distribution
    if not scores:
        # No sentiment words found
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    # Separate positive and negative scores
    pos_scores = [s for s in scores if s > 0]
    neg_scores = [s for s in scores if s < 0]
    
    # Calculate proportions
    pos_sum = sum(pos_scores)
    neg_sum = abs(sum(neg_scores))  # Make positive for proportion calculation
    
    total = pos_sum + neg_sum
    if total > 0:
        pos_prop = pos_sum / total
        neg_prop = neg_sum / total
        neu_prop = 0.0  # In this simple version, neutral is when no sentiment words
    else:
        pos_prop = neg_prop = 0.0
        neu_prop = 1.0
    
    # STEP 6: Calculate compound score (overall sentiment)
    # Compound score combines all sentiment into single [-1, 1] score
    # Used for final classification: positive if > 0.05, negative if < -0.05
    compound = (pos_sum - neg_sum) / (total + 1) if total > 0 else 0.0
    compound = max(-1, min(1, compound))  # Clamp to [-1, 1] range
    
    return {
        "positive": round(pos_prop, 3),
        "negative": round(neg_prop, 3), 
        "neutral": round(neu_prop, 3),
        "compound": round(compound, 3)
    }

def classify_sentiment(sentiment_scores: Dict[str, float]) -> str:
    """
    Convert sentiment scores to simple classification.
    
    CLASSIFICATION THRESHOLDS (VADER standard):
    - compound >= 0.05: positive
    - compound <= -0.05: negative  
    - -0.05 < compound < 0.05: neutral
    
    These thresholds are empirically determined from testing.
    """
    compound = sentiment_scores['compound']
    
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# COMPLETE INTERVIEW DEMONSTRATION
if __name__ == "__main__":
    print("RULE-BASED SENTIMENT ANALYSIS - Interview Demo")
    print("=" * 55)
    
    # Test cases that demonstrate different rules
    test_cases = [
        "This movie is great!",                    # Basic positive
        "I hate this terrible product",            # Basic negative  
        "Not bad at all",                         # Negation handling
        "Very good movie",                        # Intensifier handling
        "This is okay",                           # Neutral case
        "Really amazing performance!!!",          # Intensifier + emphasis
        "I don't really like this very much",     # Complex: negation + intensifier
    ]
    
    print("TEST CASES - Demonstrating Different Rules:")
    print("-" * 45)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Analyzing: '{text}'")
        print("   Rules applied:")
        
        # Analyze sentiment (with debug output)
        scores = analyze_sentiment(text)
        classification = classify_sentiment(scores)
        
        # Show final result
        print(f"   RESULT: {classification.upper()} (compound: {scores['compound']})")
        print(f"   Distribution: pos={scores['positive']}, neg={scores['negative']}, neu={scores['neutral']}")
    
    print(f"\n" + "=" * 55)
    print("RULE-BASED VS ML APPROACHES:")
    print("=" * 55)
    
    comparison_text = "This movie is not very good"
    
    print(f"Example: '{comparison_text}'")
    scores = analyze_sentiment(comparison_text)
    
    print(f"\nRule-based result: {classify_sentiment(scores)} ({scores['compound']})")
    print("Rule-based logic:")
    print("  1. 'good' has positive score (+2.0)")
    print("  2. 'very' intensifies it (+30% boost)")  
    print("  3. 'not' negates it (flip and dampen)")
    print("  4. Final: negative sentiment")
    
    print(f"\n" + "=" * 55)
    print("INTERVIEW TALKING POINTS:")
    print("=" * 55)
    
    print("ADVANTAGES of Rule-based:")
    print("• Fast and lightweight (no model training)")
    print("• Interpretable (can trace why decision was made)")
    print("• No training data required")
    print("• Handles grammatical rules explicitly")
    print("• Good for domain-specific customization")
    
    print(f"\nDISADVANTAGES of Rule-based:")
    print("• Limited by lexicon coverage")
    print("• Struggles with sarcasm, context")
    print("• Requires manual rule engineering")
    print("• Domain-specific performance")
    
    print(f"\nWHEN TO USE:")
    print("• Need fast, lightweight solution")
    print("• Limited or no training data")
    print("• Need explainable predictions")
    print("• Domain-specific sentiment (e.g., financial, medical)")
    
    print(f"\n" + "=" * 55)
    print("COMMON FOLLOW-UP QUESTIONS:")
    print("=" * 55)
    print("Q: How to handle sarcasm?")
    print("A: Very difficult for rule-based. Need context/ML approaches.")
    print()
    print("Q: What about domain-specific sentiment?")
    print("A: Customize lexicon for domain (financial, medical terms)")
    print()
    print("Q: Rule-based vs machine learning?")
    print("A: Rules: fast/interpretable. ML: better accuracy/context")
    print()
    print("Q: How to improve this approach?")
    print("A: Larger lexicon, better grammar rules, emoji handling")