# Problem: Rule-Based Sentiment Analysis (VADER-style)

Implement a rule-based sentiment analyzer similar to VADER:
1. `VaderSentimentAnalyzer()` - Main analyzer class
2. `analyze_sentiment(text: str) -> Dict[str, float]` - Return compound, pos, neu, neg scores
3. `score_lexicon(words: List[str]) -> float` - Score based on sentiment lexicon
4. `handle_intensifiers(words: List[str]) -> List[float]` - Boost/reduce based on intensifiers

Example:
Input: "This movie is absolutely fantastic!"
Output: {
    "compound": 0.78,
    "pos": 0.65,
    "neu": 0.25,
    "neg": 0.10
}

Requirements:
- Build sentiment lexicon with polarity scores
- Handle intensifiers ("very", "extremely", "quite")
- Process negations ("not good" → negative)
- Handle punctuation emphasis ("!!!")
- Normalize scores to [-1, 1] range

Follow-ups:
- Compare with machine learning approaches
- Handle emoji sentiment
- Domain-specific lexicon adaptation
